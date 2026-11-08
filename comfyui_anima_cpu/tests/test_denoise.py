import json
from pathlib import Path
import sys
import tempfile
import unittest
import types
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anima_cpu.core import pad_context
from anima_cpu.denoise_core import euler_trajectory, fit, load_cache, match_loss, save_cache
from anima_cpu.denoise_runtime import gradient_runtime, sampled_fingerprint
from anima_cpu.lora_train import OutputLoRA, normalize_output


class DenoiseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def examples(self):
        features = torch.randn(1, 7, 8)
        pre = torch.randn(1, 7, 8)
        base = normalize_output(pre, torch.ones(8), 1e-6)
        teacher = OutputLoRA(8, 8, 4, torch.ones(8), 1e-6, torch.Generator().manual_seed(42))
        with torch.no_grad():
            teacher.up.normal_(std=.12)
        rows = []
        for seed, split in ((1, 'train'), (2, 'validation')):
            x = torch.randn(1, 8, 1, 1)
            row = dict(features=features, pre_norm=pre, base=base, x=x, sigma=torch.tensor([.7]),
                       scene='test scene', seed=seed, split=split, step=0)
            row['baseline'] = self.predict(row, pad_context(base)).detach()
            row['target'] = self.predict(row, pad_context(teacher(row))).detach()
            rows.append(row)
        return rows

    @staticmethod
    def predict(row, context):
        return row['x'] * .1 + context[:, :7].mean(dim=1)[:, :, None, None]

    def student(self):
        return OutputLoRA(8, 8, 4, torch.ones(8), 1e-6, torch.Generator().manual_seed(42))

    def test_teacher_trajectory_uses_identical_inputs_and_euler_states(self):
        calls = []
        def predict(x, sigma, context):
            calls.append(x.clone())
            return torch.ones_like(x) * context
        rows, final = euler_trajectory(predict, torch.zeros(1, 2, 1, 1),
                                        torch.tensor([1., .5, 0.]), 2., 0.)
        torch.testing.assert_close(calls[0], calls[1])
        torch.testing.assert_close(calls[2], calls[3])
        torch.testing.assert_close(rows[1]['x'], torch.ones_like(final))
        torch.testing.assert_close(final, torch.ones_like(final) * 2)
        self.assertEqual(rows[0]['sigma'].item(), 1.)

    def test_cache_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cache.safetensors'
            rows = self.examples()
            save_cache(path, rows, {'model_fingerprint': 'test'})
            actual, info = load_cache(path)
            torch.testing.assert_close(actual[1]['target'], rows[1]['target'])
            self.assertEqual(info['model_fingerprint'], 'test')
            with self.assertRaises(FileExistsError):
                save_cache(path, rows, {})
            self.assertFalse(path.with_suffix('.lock').exists())

    def test_rejects_overlap_nonfinite_and_no_teacher_signal(self):
        rows = self.examples()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cache.safetensors'
            rows[1]['seed'] = rows[0]['seed']
            save_cache(path, rows, {})
            with self.assertRaisesRegex(ValueError, 'overlap'):
                load_cache(path)
        row = self.examples()[0]
        row['target'] = row['baseline']
        with self.assertRaisesRegex(ValueError, 'indistinguishable'):
            match_loss(row['baseline'], row)
        rows[0]['x'][0, 0, 0, 0] = float('nan')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.safetensors'
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                save_cache(path, rows, {})
            self.assertFalse(path.exists())

    def test_denoiser_training_reduces_heldout_loss(self):
        rows = self.examples()
        student = self.student()
        best, report = fit(student, rows, self.predict, 40, .01, 42)
        self.assertIsNotNone(best)
        self.assertGreater(report['first_gradient_norm'], 0)
        self.assertLess(report['best_validation_loss'], .2)
        self.assertAlmostEqual(report['initial_validation_loss'], 1., places=5)
        self.assertEqual(report['visual_validation'], 'not_performed')

    def test_detached_prediction_and_parity_fail_loudly(self):
        with self.assertRaisesRegex(RuntimeError, 'detached'):
            fit(self.student(), self.examples(), lambda r, c: self.predict(r, c).detach(), 1, .001, 42)
        with self.assertRaisesRegex(ValueError, 'parity'):
            fit(self.student(), self.examples(), lambda r, c: self.predict(r, c) + 1, 1, .001, 42)

    def test_cancellation_propagates(self):
        def cancel(value):
            raise InterruptedError('cancelled')
        with self.assertRaises(InterruptedError):
            fit(self.student(), self.examples(), self.predict, 10, .001, 42, cancel)

    def test_inference_weights_checkpoint_gradient_and_restore_on_failure(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)])
                self.register_buffer('scale', torch.ones(8))
            def forward(self, x):
                for block in self.blocks:
                    x = block(x).tanh()
                return x * self.scale
        with torch.inference_mode():
            model = torch.nn.Module()
            model.diffusion_model = Tiny().requires_grad_(False)
        original = model.diffusion_model.blocks[0].weight
        before = sampled_fingerprint(model)
        x = torch.randn(2, 8, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with gradient_runtime(model), torch.autograd.graph.save_on_cpu():
                self.assertFalse(torch.is_inference(model.diffusion_model.blocks[0].weight))
                model.diffusion_model(x).square().mean().backward()
                self.assertGreater(x.grad.abs().sum().item(), 0)
                raise RuntimeError('intentional')
        self.assertIs(model.diffusion_model.blocks[0].weight, original)
        self.assertNotIn('forward', model.diffusion_model.blocks[0].__dict__)
        self.assertEqual(sampled_fingerprint(model), before)

    def test_checkpoint_matches_unwrapped_gradient(self):
        model = torch.nn.Module()
        model.diffusion_model = torch.nn.Module()
        model.diffusion_model.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
        model.requires_grad_(False)
        x = torch.randn(2, 8, requires_grad=True)
        model.diffusion_model.blocks[0](x).square().sum().backward()
        expected = x.grad.clone()
        x.grad = None
        with gradient_runtime(model):
            model.diffusion_model.blocks[0](x).square().sum().backward()
        torch.testing.assert_close(x.grad, expected)


    def test_train_bridge_under_comfy_inference_mode(self):
        from anima_cpu.denoise_runtime import train_cache, model_signature
        class Base(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.diffusion_model = torch.nn.Module()
                self.diffusion_model.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
                self.diffusion_model.llm_adapter = torch.nn.Module()
                self.diffusion_model.llm_adapter.out_proj = torch.nn.Linear(8, 8)
                self.diffusion_model.llm_adapter.norm = torch.nn.RMSNorm(8, eps=1e-6)
                self.model_sampling = torch.nn.Identity()
                self.requires_grad_(False)
            def get_dtype_inference(self):
                return torch.float32
            def apply_model(self, x, sigma, c_crossattn, transformer_options):
                h = c_crossattn[:, :7].mean(dim=1)
                h = self.diffusion_model.blocks[0](h)
                return x * .1 + h[:, :, None, None]
        with torch.inference_mode():
            base = Base()
            adapter = {'module': base.diffusion_model.llm_adapter, 'sha256': 'test-sha'}
            rows = self.examples()
            teacher = self.student()
            teacher.up.normal_(std=.1)
            for row in rows:
                row['baseline'] = base.apply_model(row['x'], row['sigma'], pad_context(row['base']), {})
                row['target'] = base.apply_model(row['x'], row['sigma'], pad_context(teacher(row)), {})
            info = dict(model_signature(base), adapter_sha256='test-sha')
        before = sampled_fingerprint(base)
        mm = types.ModuleType('comfy.model_management')
        mm.throw_exception_if_processing_interrupted = lambda: None
        with patch.dict(sys.modules, {'comfy.model_management': mm}), \
             patch('anima_cpu.denoise_runtime.load_model', return_value=(base, torch.device('cpu'))), \
             torch.inference_mode():
            best, report = train_cache(None, adapter, rows, info, 10, 4, .01, 42, lambda v: None)
        self.assertGreater(report['first_gradient_norm'], 0)
        self.assertIsNotNone(best)
        self.assertEqual(sampled_fingerprint(base), before)
        self.assertEqual(report['device'], 'cpu')
        self.assertNotIn('forward', base.diffusion_model.blocks[0].__dict__)

    def test_example_workflow_links_are_consistent(self):
        for path in (Path(__file__).resolve().parents[1] / 'examples').glob('denoise_*workflow.json'):
            workflow = json.loads(path.read_text())
            nodes = {n['id']: n for n in workflow['nodes']}
            for ident, source, slot, dest, input_slot, typ in workflow['links']:
                self.assertEqual(nodes[dest]['inputs'][input_slot]['link'], ident)
                self.assertEqual(nodes[dest]['inputs'][input_slot]['type'], typ)
                self.assertIn(ident, nodes[source]['outputs'][slot]['links'])
                self.assertEqual(nodes[source]['outputs'][slot]['type'], typ)


if __name__ == '__main__':
    unittest.main()
