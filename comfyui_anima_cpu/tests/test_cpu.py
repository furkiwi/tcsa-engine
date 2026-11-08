import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import gguf
from gguf.quants import quantize, dequantize
import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anima_cpu.core import (load_profile, pad_context, prefix_length, profile_path,
                            save_profile, transplant)
from anima_cpu.backend import read_weights
from compare_api import prepare


class CoreTests(unittest.TestCase):
    def test_zero_and_strength_endpoints(self):
        base = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
        prefix = torch.ones(2, 8)
        self.assertTrue(torch.equal(transplant(base, prefix, 0), pad_context(base)))
        expected = pad_context(torch.cat((prefix[None], base), dim=1))
        self.assertTrue(torch.equal(transplant(base, prefix, 1), expected))
        self.assertTrue(torch.equal(transplant(base, prefix, .5), (pad_context(base) + expected) / 2))

    def test_invalid_shapes_strength_and_overflow(self):
        for strength in (float('nan'), float('inf'), -1, 2):
            with self.assertRaises(ValueError):
                transplant(torch.zeros(1, 2, 8), torch.zeros(1, 8), strength)
        with self.assertRaises(ValueError):
            transplant(torch.zeros(1, 512, 8), torch.zeros(1, 8), 1)
        with self.assertRaises(ValueError):
            transplant(torch.zeros(1, 2, 8), torch.zeros(1, 16), 1)

    def test_alignment(self):
        self.assertEqual(prefix_length([4, 5, 1], [9, 8, 4, 5, 1]), 2)
        with self.assertRaises(ValueError):
            prefix_length([4, 5, 1], [9, 8, 7, 5, 1])

    def test_profile_roundtrip_overwrite_lock_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = profile_path(directory, "trial")
            values = torch.ones(2, 1024)
            save_profile(path, values, values, {"artist": "@test"})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_profile(path, values * 2, values, {})
            self.assertEqual(path.read_bytes(), original)
            save_profile(path, values * 2, values, {"artist": "@new"}, True)
            prefix, _, info = load_profile(path)
            self.assertTrue(torch.equal(prefix, values * 2))
            self.assertEqual(info["artist"], "@new")
            path.with_suffix('.lock').touch()
            with self.assertRaises(FileExistsError):
                save_profile(path, values, values, {}, True)
            with self.assertRaises(ValueError):
                profile_path(directory, "../escape")

    def test_nonfinite_profile_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            save_profile(path, torch.full((1, 1024), float('nan')), torch.zeros(1, 1024), {})
            with self.assertRaises(ValueError):
                load_profile(path)

    def test_safetensors_filters_diffusion_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file({"llm_adapter.embed.weight": torch.ones(2, 4),
                       "diffusion.large.weight": torch.zeros(4, 4)}, str(path))
            weights = dict(read_weights(path))
            self.assertEqual(list(weights), ["embed.weight"])
            self.assertTrue(torch.equal(weights["embed.weight"], torch.ones(2, 4)))

    def test_q3_k_gguf_adapter_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'q3.gguf'
            writer = gguf.GGUFWriter(str(path), 'cosmos')
            qtype = gguf.GGMLQuantizationType.Q3_K
            block, size = gguf.GGML_QUANT_SIZES[qtype]
            packed = np.zeros((2, size), dtype=np.uint8)
            writer.add_tensor('llm_adapter.test.weight', packed, raw_dtype=qtype)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            loaded = dict(read_weights(path))['test.weight']
            self.assertEqual(tuple(loaded.shape), (2, block))
            self.assertTrue(torch.equal(loaded, torch.zeros_like(loaded)))

    def test_api_comparison_preserves_sampling_and_input(self):
        graph = {'1': {'class_type': 'TCSAAnimaCompare', 'inputs': {'mode': 'extracted'}},
                 '2': {'class_type': 'KSampler', 'inputs': {'positive': ['1', 0], 'seed': 8, 'steps': 4, 'cfg': 1}},
                 '3': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'old'}}}
        prepared = prepare(graph, 'native_artist', 42)
        self.assertEqual(prepared['2']['inputs']['seed'], 42)
        self.assertEqual(prepared['2']['inputs']['steps'], 4)
        self.assertEqual(prepared['2']['inputs']['cfg'], 1)
        self.assertEqual(graph['2']['inputs']['seed'], 8)
        self.assertEqual(prepared['1']['inputs']['mode'], 'native_artist')

    def test_real_gguf_q4_f16_and_original_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            writer = gguf.GGUFWriter(str(path), "cosmos")
            rng = np.random.default_rng(4)
            source = rng.normal(size=(2, 256)).astype(np.float32)
            qtype = gguf.GGMLQuantizationType.Q4_0
            packed = quantize(source, qtype)
            name = "model.diffusion_model.llm_adapter.test.weight"
            writer.add_tensor(name, packed, raw_dtype=qtype)
            writer.add_array("comfy.gguf.orig_shape." + name, [1, 2, 256])
            writer.add_tensor("llm_adapter.bias", np.ones(8, dtype=np.float16))
            writer.add_tensor("diffusion.ignored", source)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            loaded = dict(read_weights(path))
            self.assertEqual(set(loaded), {"test.weight", "bias"})
            np.testing.assert_array_equal(loaded["test.weight"].numpy(), dequantize(packed, qtype).reshape(1, 2, 256))
            self.assertEqual(loaded["bias"].dtype, torch.float32)


class NodeTests(unittest.TestCase):
    def setUp(self):
        folders = types.ModuleType("folder_paths")
        folders.folder_names_and_paths = {}
        self.modules = patch.dict(sys.modules, {"folder_paths": folders})
        self.modules.start()
        self.nodes = importlib.import_module("anima_cpu.nodes")
        self.base = torch.ones(1, 3, 1024)
        self.prefix = torch.ones(2, 1024) * 3
        self.probe = torch.ones(4, 1024)
        self.adapter = {"sha256": "hash", "checkpoint": "test.gguf"}
        self.profile = (self.prefix, self.probe, {"scenes": ["training scene"], "artist": "@test",
                                                "adapter_sha256": "hash", "prefix_ids": [7, 8],
                                                "checkpoint": "test.gguf"})

    def tearDown(self):
        self.modules.stop()

    def encode(self, adapter, clip, text, allow_weighted=False):
        ids, value = ([7, 8, 2, 3, 1], torch.cat((self.prefix[None], self.base), dim=1)) if text.startswith('@') else ([2, 3, 1], self.base)
        return ids, value, {"cond": self.base, "t5xxl_ids": torch.tensor(ids),
                            "t5xxl_weights": torch.ones(len(ids)), "attention_mask": torch.ones(1, 3)}

    def test_compare_modes_metadata_and_zero(self):
        with patch.object(self.nodes, "encode", side_effect=self.encode), patch.object(self.nodes, "encoder_probe", return_value=self.probe):
            for mode in self.nodes.MODES:
                result = self.nodes.Compare().run(self.adapter, None, self.profile, "new scene", mode, 1)
                cond, report = result['result']
                self.assertEqual(json.loads(report)['zero_max_abs_error'], 0)
                if mode.startswith('native'):
                    self.assertIn('t5xxl_ids', cond[0][1])
                else:
                    self.assertEqual(cond[0][1], {})
                    self.assertEqual(cond[0][0].shape, (1, 512, 1024))
                if mode in ('zero_strength', 'cpu_roundtrip'):
                    self.assertTrue(torch.equal(cond[0][0], pad_context(self.base)))

    def test_wrong_adapter_and_training_scene_rejected(self):
        with self.assertRaises(ValueError):
            self.nodes.Compare().run({"sha256": "wrong"}, None, self.profile, "new scene", "extracted", 1)
        with self.assertRaises(ValueError):
            self.nodes.Compare().run(self.adapter, None, self.profile, "training scene", "extracted", 1)

    def test_wrong_encoder_rejected(self):
        with patch.object(self.nodes, "encoder_probe", return_value=self.probe * 10):
            with self.assertRaises(ValueError):
                self.nodes.Compare().run(self.adapter, None, self.profile, "new scene", "extracted", 1)


if __name__ == '__main__':
    unittest.main()
