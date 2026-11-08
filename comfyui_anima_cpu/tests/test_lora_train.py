import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anima_cpu.core import pad_context
from anima_cpu.lora_train import (attention_response, export_lora, fit_lora,
                                  normalize_output, train_profile)


class LoRATests(unittest.TestCase):
    def samples(self):
        generator = torch.Generator().manual_seed(5)
        weight = torch.randn(8, 8, generator=generator) / 3
        delta = torch.randn(8, 1, generator=generator) @ torch.randn(1, 8, generator=generator) * .1
        samples = []
        for _ in range(6):
            features = torch.randn(1, 5, 8, generator=generator)
            before = F.linear(features, weight)
            base = normalize_output(before, torch.ones(8), 1e-6)
            target_context = normalize_output(F.linear(features, weight + delta), torch.ones(8), 1e-6)
            queries = torch.randn(12, 8, generator=generator) * 3
            samples.append({'features': features, 'pre_norm': before, 'base': base,
                            'queries': queries, 'target': attention_response(pad_context(target_context), queries)})
        return samples, weight

    def test_training_under_inference_mode_and_exported_patch(self):
        # ComfyUI wraps node execution in inference mode. Training must override it.
        with torch.inference_mode():
            samples, weight = self.samples()
            (down, up), report = fit_lora(samples[:4], samples[4:], torch.ones(8), 1e-6,
                                          rank=2, steps=80, learning_rate=.02, seed=42)
        self.assertGreater(report['selected_step'], 0)
        self.assertLess(report['best_validation_loss'], report['initial_validation_loss'])
        self.assertGreater(up.abs().max().item(), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.safetensors'
            export_lora(path, down, up, report)
            tensors = load_file(str(path))
            prefix = 'diffusion_model.llm_adapter.out_proj'
            rank = down.shape[0]
            merged = weight + (tensors[prefix+'.lora_up.weight'] @ tensors[prefix+'.lora_down.weight']) * tensors[prefix+'.alpha'] / rank
            sample = samples[-1]
            direct = normalize_output(F.linear(sample['features'], merged), torch.ones(8), 1e-6)
            trained = normalize_output(sample['pre_norm'] + F.linear(F.linear(sample['features'], down), up), torch.ones(8), 1e-6)
            self.assertTrue(torch.allclose(direct, trained, atol=1e-6))
            with safe_open(str(path), framework='pt') as file:
                self.assertEqual(json.loads(file.metadata()['info'])['visual_validation'], 'not_performed')
            with self.assertRaises(FileExistsError):
                export_lora(path, down, up, report)

    def test_zero_target_keeps_initial_weights(self):
        samples, _ = self.samples()
        for sample in samples:
            sample['target'] = attention_response(pad_context(sample['base']), sample['queries'])
        (_, up), report = fit_lora(samples[:4], samples[4:], torch.ones(8), 1e-6, 2, 20, .001, 42)
        self.assertEqual(report['selected_step'], 0)
        self.assertEqual(up.abs().max().item(), 0)

    def test_normalization_mismatch_rejected(self):
        samples, _ = self.samples()
        with self.assertRaises(ValueError):
            fit_lora(samples[:4], samples[4:], torch.ones(8) * 2, 1e-6, 2, 20, .001, 42)

    def test_cancellation_propagates(self):
        samples, _ = self.samples()
        def interrupt(step, total):
            raise InterruptedError('cancelled')
        with self.assertRaises(InterruptedError):
            fit_lora(samples[:4], samples[4:], torch.ones(8), 1e-6, 2, 20, .001, 42, interrupt)

    def test_profile_training_caches_teacher_and_does_not_mutate_adapter(self):
        class TinyAdapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.out_proj = torch.nn.Linear(8, 8)
                self.norm = torch.nn.RMSNorm(8, eps=1e-6)
                with torch.no_grad():
                    self.out_proj.weight.copy_(torch.eye(8))
                    self.out_proj.bias.zero_()

        module = TinyAdapter().requires_grad_(False)
        original = {k: v.clone() for k, v in module.state_dict().items()}
        adapter = {'module': module, 'sha256': 'fixture'}
        probe = torch.ones(2, 8)
        profile = (torch.ones(2, 8), probe, {'artist': '@test', 'checkpoint': 'test.gguf',
                                           'adapter_sha256': 'fixture', 'scenes': ['old']})
        def fake_encode(adapter, clip, scene):
            generator = torch.Generator().manual_seed(sum(map(ord, scene)))
            features = torch.randn(1, 4, 8, generator=generator)
            return [1, 2, 3, 4], module.norm(module.out_proj(features)), {}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trained.safetensors'
            with patch('anima_cpu.lora_train.encode', side_effect=fake_encode), patch('anima_cpu.lora_train.encoder_probe', return_value=probe):
                with torch.inference_mode():
                    report = train_profile(adapter, None, profile, 'one\ntwo\nthree\nfour', 'five\nsix',
                                           2, 80, .02, 42, path)
            self.assertTrue(path.exists())
            self.assertEqual(report['preservation_anchor'], 'empty_prompt')
            self.assertEqual(len(module.out_proj._forward_hooks), 0)
            for key, value in module.state_dict().items():
                self.assertTrue(torch.equal(value, original[key]))


if __name__ == '__main__':
    unittest.main()
