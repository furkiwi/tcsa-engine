"""Regression coverage for weighted artist extraction through public node entry points."""
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anima_cpu.backend import encode, raw_encode
from anima_cpu.core import load_profile, pad_context

ARTISTS = r"(@sunapua:0.8), (@waonaolmao:0.9), (@oka \(bananashoe\):0.7)"


class FakeClip:
    """Fixed token fixtures isolate propagation from ComfyUI's tokenizer implementation."""
    def __init__(self):
        self.patcher = types.SimpleNamespace(load_device=torch.device('cpu'), patches={})
        self.seen = []
        self.override_weights = None

    def tokenize(self, text):
        self.seen.append(text)
        return text

    def encode_from_tokens(self, text, **kwargs):
        ids, weights = [2, 3, 1], [1., 1., 1.]
        if text.startswith(ARTISTS):
            ids = [7, 8, 9] + ids
            weights = [.8, .9, .7] + weights
        if self.override_weights is not None:
            weights = self.override_weights
        return {'cond': torch.ones(1, 4, 1024), 't5xxl_ids': torch.tensor(ids),
                't5xxl_weights': torch.tensor(weights), 'pooled_output': None}


def fake_adapter(hidden, ids):
    return ids.float().unsqueeze(-1).expand(-1, -1, 1024)


class WeightedArtistTests(unittest.TestCase):
    def setUp(self):
        self.clip = FakeClip()
        self.adapter = {'module': fake_adapter, 'sha256': 'fixture', 'checkpoint': 'fixture.gguf'}

    def test_weighted_native_equivalence_and_escaped_input(self):
        ids, projected, raw = encode(self.adapter, self.clip, ARTISTS + ', a fox', allow_weighted=True)
        native = fake_adapter(raw['cond'], torch.tensor(ids)[None]) * raw['t5xxl_weights'][None, :, None]
        self.assertTrue(torch.allclose(projected, native, rtol=0, atol=0))
        self.assertTrue(torch.allclose(projected[0, :3, 0], torch.tensor([7*.8, 8*.9, 9*.7])))
        self.assertEqual(self.clip.seen, [ARTISTS + ', a fox'])
        self.assertTrue(torch.allclose(raw['t5xxl_weights'][:3], torch.tensor([.8, .9, .7])))

    def test_unweighted_regression_and_scene_guard(self):
        ids, projected, raw = encode(self.adapter, self.clip, 'a fox')
        self.assertTrue(torch.allclose(projected, fake_adapter(raw['cond'], torch.tensor(ids)[None]), rtol=0, atol=0))
        with self.assertRaises(ValueError):
            encode(self.adapter, self.clip, ARTISTS + ', a fox')

    def test_zero_weight_and_invalid_weights(self):
        self.clip.override_weights = [0., 1., 1.]
        _, projected, _ = encode(self.adapter, self.clip, 'a fox', allow_weighted=True)
        self.assertEqual(projected[0, 0].abs().max().item(), 0)
        for invalid in ([float('nan'), 1., 1.], [float('inf'), 1., 1.], [.5]):
            self.clip.override_weights = invalid
            with self.assertRaises(ValueError):
                raw_encode(self.clip, 'a fox', allow_weighted=True)

    def test_extract_save_reload_compare_weighted_artists(self):
        folders = types.ModuleType('folder_paths')
        utils = types.ModuleType('comfy.utils')
        utils.ProgressBar = lambda size: types.SimpleNamespace(update_absolute=lambda n: None)
        with patch.dict(sys.modules, {'folder_paths': folders, 'comfy.utils': utils}):
            nodes = importlib.import_module('anima_cpu.nodes')
            with tempfile.TemporaryDirectory() as directory, patch.object(nodes, 'output_directory', return_value=Path(directory)):
                result = nodes.Extract().run(self.adapter, self.clip, ARTISTS, 'a fox\na harbor', 'mix')
                profile = load_profile(Path(directory) / 'mix.safetensors')
                self.assertEqual(profile[2]['artist'], ARTISTS)
                self.assertTrue(torch.allclose(profile[0][:, 0], torch.tensor([7*.8, 8*.9, 9*.7])))
                self.assertTrue(torch.allclose(result['result'][0][0], profile[0]))
                native = nodes.Compare().run(self.adapter, self.clip, profile, 'a robot', 'native_artist', 1)['result'][0]
                self.assertTrue(torch.allclose(native[0][1]['t5xxl_weights'][:3], torch.tensor([.8, .9, .7])))
                extracted = nodes.Compare().run(self.adapter, self.clip, profile, 'a robot', 'extracted', 1)['result'][0]
                self.assertEqual(extracted[0][1], {})
                self.assertTrue(torch.allclose(extracted[0][0][0, :3], profile[0], rtol=0, atol=0))
                zero = nodes.Compare().run(self.adapter, self.clip, profile, 'a robot', 'zero_strength', 1)['result'][0]
                _, base, _ = encode(self.adapter, self.clip, 'a robot')
                self.assertTrue(torch.allclose(zero[0][0], pad_context(base), rtol=0, atol=0))


if __name__ == '__main__':
    unittest.main()
