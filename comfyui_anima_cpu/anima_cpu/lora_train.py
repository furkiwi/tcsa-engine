"""Experimental CPU distillation of prefix conditioning into out_proj LoRA.

The objective matches attention responses in conditioning space, not DiT noise
predictions. Image-level fidelity must be checked separately after loading.
"""
import json
import math
import os
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .backend import encode, encoder_probe
from .core import pad_context, transplant


def normalize_output(value, weight, eps):
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps) * weight


def attention_response(context, queries):
    logits = queries @ context[0].T / math.sqrt(context.shape[-1])
    return logits.softmax(dim=-1) @ context[0]


def make_queries(prefix, base, generator):
    def pick(value):
        indices = torch.linspace(0, value.shape[0] - 1, min(8, value.shape[0])).long()
        return value[indices]
    candidates = torch.cat((pick(prefix), pick(base[0]),
                            torch.randn(16, base.shape[-1], generator=generator)))
    return F.normalize(candidates, dim=-1) * math.sqrt(base.shape[-1])


@torch.inference_mode()
def cache_example(adapter, clip, prefix, scene, generator):
    captured = {}

    def capture(module, args, output):
        captured['features'] = args[0].detach().cpu().float()
        captured['pre_norm'] = output.detach().cpu().float()

    handle = adapter['module'].out_proj.register_forward_hook(capture)
    try:
        _, base, _ = encode(adapter, clip, scene)
    finally:
        handle.remove()
    queries = make_queries(prefix, base, generator)
    teacher = transplant(base, prefix, 1.0)
    return {**captured, 'base': base, 'queries': queries,
            'target': attention_response(teacher, queries)}


class OutputLoRA(torch.nn.Module):
    def __init__(self, input_dim, output_dim, rank, norm_weight, eps, generator):
        super().__init__()
        self.down = torch.nn.Parameter(torch.randn(rank, input_dim, generator=generator) / math.sqrt(input_dim))
        self.up = torch.nn.Parameter(torch.zeros(output_dim, rank))
        self.register_buffer('norm_weight', norm_weight.detach().clone())
        self.eps = eps

    def forward(self, sample):
        delta = F.linear(F.linear(sample['features'], self.down), self.up)
        return normalize_output(sample['pre_norm'] + delta, self.norm_weight, self.eps)

    def objective(self, sample):
        output = self(sample)
        response = attention_response(pad_context(output), sample['queries'])
        match = F.mse_loss(response, sample['target']) / sample['target'].square().mean().clamp_min(1e-8)
        drift = F.mse_loss(output, sample['base']) / sample['base'].square().mean().clamp_min(1e-8)
        return match + 0.05 * drift


def fit_lora(train, validation, norm_weight, eps, rank, steps, learning_rate, seed, progress=None):
    """Train ordinary CPU tensors even when called by ComfyUI inference mode."""
    with torch.inference_mode(False), torch.enable_grad():
        train = [{key: value.detach().clone() for key, value in sample.items()} for sample in train]
        validation = [{key: value.detach().clone() for key, value in sample.items()} for sample in validation]
        generator = torch.Generator(device='cpu').manual_seed(seed)
        model = OutputLoRA(train[0]['features'].shape[-1], train[0]['base'].shape[-1],
                           rank, norm_weight, eps, generator)
        for sample in train + validation:
            if not torch.allclose(model(sample), sample['base'], rtol=1e-4, atol=1e-5):
                raise ValueError('CPU output normalization does not match this Anima adapter.')

        def evaluate(samples):
            with torch.no_grad():
                return sum(model.objective(sample).item() for sample in samples) / len(samples)

        initial = evaluate(validation)
        if not math.isfinite(initial):
            raise ValueError('Non-finite initial loss.')
        best_loss, best_step = initial, 0
        best = (model.down.detach().clone(), model.up.detach().clone())
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
        history = [{'step': 0, 'validation_loss': initial}]
        order = torch.randperm(len(train), generator=generator).tolist()
        for step in range(1, steps + 1):
            if (step - 1) % len(train) == 0:
                order = torch.randperm(len(train), generator=generator).tolist()
            sample = train[order[(step - 1) % len(train)]]
            optimizer.zero_grad(set_to_none=True)
            loss = model.objective(sample)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss. Try a lower learning rate.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step % 20 == 0 or step == steps:
                value = evaluate(validation)
                if not math.isfinite(value):
                    raise ValueError('Non-finite validation loss.')
                history.append({'step': step, 'validation_loss': value})
                if value < best_loss:
                    best_loss, best_step = value, step
                    best = (model.down.detach().clone(), model.up.detach().clone())
            if progress is not None:
                progress(step, steps)
        return best, {'initial_validation_loss': initial, 'best_validation_loss': best_loss,
                      'selected_step': best_step, 'history': history,
                      'objective': 'conditioning_attention_response_plus_0.05_content_drift',
                      'visual_validation': 'not_performed'}


def export_lora(path, down, up, info, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix('.lock')
    with lock.open('x'):
        pass
    temporary = None
    try:
        if path.exists() and not overwrite:
            raise FileExistsError('LoRA exists. Use a new name or enable overwrite.')
        prefix = 'diffusion_model.llm_adapter.out_proj'
        tensors = {prefix + '.lora_down.weight': down.detach().cpu().float().contiguous(),
                   prefix + '.lora_up.weight': up.detach().cpu().float().contiguous(),
                   prefix + '.alpha': torch.tensor(float(down.shape[0]))}
        if not all(torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError('Non-finite LoRA tensors.')
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
        save_file(tensors, str(temporary), metadata={
            'format': 'tcsa-anima-out-proj-lora-v1', 'ss_network_dim': str(down.shape[0]),
            'ss_network_alpha': str(down.shape[0]), 'info': json.dumps(info, ensure_ascii=False)})
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def train_profile(adapter, clip, profile, train_scenes, validation_scenes, rank,
                  steps, learning_rate, seed, path, overwrite=False, progress=None):
    prefix, probe, info = profile
    if info['adapter_sha256'] != adapter['sha256']:
        raise ValueError('Use the same adapter checkpoint as the successful profile.')
    actual = encoder_probe(clip)
    if actual.shape != probe.shape or not torch.allclose(actual, probe, rtol=1e-4, atol=1e-5):
        raise ValueError('Use the same CPU text encoder as the successful profile.')
    train_texts = list(dict.fromkeys(x.strip() for x in train_scenes.splitlines() if x.strip()))
    val_texts = list(dict.fromkeys(x.strip() for x in validation_scenes.splitlines() if x.strip()))
    if not 4 <= len(train_texts) <= 64 or not 2 <= len(val_texts) <= 16:
        raise ValueError('Provide 4–64 training scenes and 2–16 validation scenes.')
    if set(train_texts) & set(val_texts) or set(info['scenes']) & set(val_texts):
        raise ValueError('Validation scenes must differ from training and profile extraction scenes.')
    if any('@' in text for text in train_texts + val_texts):
        raise ValueError('Training and validation scenes must not contain artist tags.')
    if not 1 <= rank <= 128 or not 1 <= steps <= 10000 or not math.isfinite(learning_rate) or not 0 < learning_rate <= .1:
        raise ValueError('Invalid rank, steps or learning rate.')
    if Path(path).exists() and not overwrite:
        raise FileExistsError('LoRA exists. Choose a new name or enable overwrite.')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    cached = []
    # LoRA also affects the unconditional branch. Anchor the empty prompt to
    # its original response while fitting the styled positive examples.
    cache_texts = train_texts + [''] + val_texts
    for index, text in enumerate(cache_texts):
        if progress:
            progress('cache', index, len(cache_texts))
        sample = cache_example(adapter, clip, prefix, text, generator)
        if text == '':
            sample['target'] = attention_response(pad_context(sample['base']), sample['queries'])
        cached.append(sample)
    norm = adapter['module'].norm
    split = len(train_texts) + 1
    best, report = fit_lora(cached[:split], cached[split:],
                            norm.weight, norm.eps, rank, steps, learning_rate, seed,
                            progress=(lambda step, total: progress('train', step, total)) if progress else None)
    if report['selected_step'] == 0:
        raise RuntimeError('Validation did not improve. No LoRA exported; try more varied training scenes.')
    metadata = {**report, 'artist': info['artist'], 'checkpoint': info['checkpoint'],
                'adapter_sha256': adapter['sha256'], 'train_scenes': train_texts,
                'validation_scenes': val_texts, 'seed': seed, 'rank': rank,
                'steps': steps, 'learning_rate': learning_rate,
                'teacher': 'successful_extracted_profile_strength_1',
                'preservation_anchor': 'empty_prompt'}
    export_lora(path, *best, metadata, overwrite)
    return {**metadata, 'saved': str(path)}
