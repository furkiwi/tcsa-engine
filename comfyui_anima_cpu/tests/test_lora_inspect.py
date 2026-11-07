from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anima_cpu.lora_inspect import inspect_lora
from anima_cpu.lora_train import export_lora
from anima_cpu.core import pad_context


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out_proj = torch.nn.Linear(8, 8, bias=False)
        self.norm = torch.nn.RMSNorm(8, eps=1e-6)
        with torch.no_grad():
            self.out_proj.weight.copy_(torch.eye(8))

    def forward(self, hidden, ids):
        return self.norm(self.out_proj(hidden))


class InspectTests(unittest.TestCase):
    def test_explicit_application_matches_merged_weights_and_restores_model(self):
        module = TinyAdapter().requires_grad_(False)
        adapter = {'module': module, 'sha256': 'fixture'}
        hidden = torch.arange(24).reshape(1, 3, 8).float() / 10
        ids = torch.tensor([1, 2, 3])
        fixture = (ids, hidden, {'t5xxl_weights': torch.ones(3)})
        down = torch.ones(2, 8) * .1
        up = torch.arange(16).reshape(8, 2).float() * .02
        original = module.out_proj.weight.clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.safetensors'
            export_lora(path, down, up, {'adapter_sha256': 'fixture'})
            with patch('anima_cpu.lora_inspect.raw_encode', return_value=fixture):
                base, applied, report = inspect_lora(adapter, None, path, 'scene', 1)
                with torch.no_grad():
                    expected = pad_context(module.norm(torch.nn.functional.linear(hidden, original + up @ down)))
                self.assertTrue(torch.allclose(applied, expected, atol=1e-6))
                self.assertGreater(report['conditioning_relative_l2'], 0)
                self.assertTrue(torch.equal(module.out_proj.weight, original))
                self.assertEqual(len(module.out_proj._forward_hooks), 0)
                base, zero, report = inspect_lora(adapter, None, path, 'scene', 0)
                self.assertTrue(torch.equal(base, zero))
                self.assertEqual(report['conditioning_relative_l2'], 0)
                with self.assertRaises(ValueError):
                    inspect_lora({'module': module, 'sha256': 'wrong'}, None, path, 'scene', 1)


if __name__ == '__main__':
    unittest.main()
