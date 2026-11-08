"""Measure the exported LoRA on a CPU adapter independently of GGUF patching."""
import json
import math

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .backend import raw_encode
from .core import pad_context


@torch.inference_mode()
def inspect_lora(adapter, clip, path, scene, strength):
    if not math.isfinite(strength):
        raise ValueError('Strength must be finite.')
    layer = adapter['module'].out_proj
    prefix = 'diffusion_model.llm_adapter.out_proj'
    with safe_open(str(path), framework='pt', device='cpu') as file:
        metadata = file.metadata() or {}
        if metadata.get('format') != 'tcsa-anima-out-proj-lora-v1':
            raise ValueError('Select a LoRA exported by TCSA Train LoRA (CPU).')
        info = json.loads(metadata['info'])
        if info['adapter_sha256'] != adapter['sha256']:
            raise ValueError('LoRA and selected adapter were made from different checkpoint weights.')
        shapes = [file.get_slice(prefix + key).get_shape() for key in ('.lora_down.weight', '.lora_up.weight')]
        down_shape, up_shape = shapes
        if (len(down_shape) != 2 or len(up_shape) != 2 or not 1 <= down_shape[0] <= 128
                or down_shape[1] != layer.weight.shape[1]
                or up_shape != [layer.weight.shape[0], down_shape[0]]):
            raise ValueError('LoRA dimensions do not match this adapter.')
        down = file.get_tensor(prefix + '.lora_down.weight').float()
        up = file.get_tensor(prefix + '.lora_up.weight').float()
        alpha = file.get_tensor(prefix + '.alpha').item()
    if not math.isfinite(alpha) or not torch.isfinite(up).all() or not torch.isfinite(down).all():
        raise ValueError('Non-finite LoRA weights.')
    scale = strength * alpha / down.shape[0]
    ids, hidden, raw = raw_encode(clip, scene)
    module = adapter['module']
    before = module(hidden, ids[None]).detach().float()
    calls = []

    def add_lora(module, args, output):
        calls.append(True)
        return output + F.linear(F.linear(args[0], down), up) * scale

    hook = layer.register_forward_hook(add_lora)
    try:
        after = module(hidden, ids[None]).detach().float()
    finally:
        hook.remove()
    if len(calls) != 1:
        raise RuntimeError('Expected exactly one out_proj invocation.')
    weights = raw['t5xxl_weights'].to(before).reshape(1, -1, 1)
    before, after = before * weights, after * weights
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError('Adapter produced non-finite conditioning.')
    delta = (up @ down) * scale
    initial = info.get('initial_validation_loss')
    best = info.get('best_validation_loss')
    report = {
        'artist': info.get('artist'), 'checkpoint': info.get('checkpoint'),
        'rank': down.shape[0], 'alpha': alpha, 'strength': strength,
        'weight_delta_relative_l2': (delta.norm() / layer.weight.float().norm().clamp_min(1e-12)).item(),
        'conditioning_relative_l2': ((after-before).norm() / before.norm().clamp_min(1e-12)).item(),
        'conditioning_max_abs_change': (after-before).abs().max().item(),
        'conditioning_cosine': F.cosine_similarity(before.flatten(), after.flatten(), dim=0).item(),
        'initial_validation_loss': initial, 'best_validation_loss': best,
        'selected_step': info.get('selected_step'),
        'note': 'Explicit CPU application. Nonzero changes do not prove style transfer or GGUF runtime patching.'}
    return pad_context(before), pad_context(after), report
