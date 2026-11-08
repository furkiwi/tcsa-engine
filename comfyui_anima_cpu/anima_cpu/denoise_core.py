"""Single-step teacher distillation. No ComfyUI imports in the numerical core."""
import json
import math
import os
from pathlib import Path
import tempfile
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = 'tcsa-anima-denoise-cache-v1'


def finite(value, label):
    if not torch.isfinite(value).all():
        raise ValueError('Non-finite ' + label)
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_cache(path, rows, info):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix('.lock')
    with lock.open('x'):
        pass
    temporary = None
    try:
        if path.exists():
            raise FileExistsError('Cache exists. Use a new cache name.')
        tensors = {}
        descriptors = []
        for i, row in enumerate(rows):
            descriptors.append({key: row[key] for key in ('scene', 'seed', 'step', 'split')})
            for key in ('x', 'sigma', 'target', 'baseline', 'features', 'pre_norm', 'base'):
                value = finite(row[key], key).detach().cpu().float().contiguous().clone()
                tensors[f'{i}.{key}'] = value
        metadata = dict(info, rows=descriptors)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
        save_file(tensors, str(temporary), metadata={'format': FORMAT, 'info': json.dumps(metadata)})
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def load_cache(path):
    with safe_open(str(path), framework='pt', device='cpu') as file:
        meta = file.metadata() or {}
        if meta.get('format') != FORMAT:
            raise ValueError('Not a single-step denoising cache.')
        info = json.loads(meta['info'])
        rows = []
        for i, description in enumerate(info['rows']):
            row = dict(description)
            for key in ('x', 'sigma', 'target', 'baseline', 'features', 'pre_norm', 'base'):
                row[key] = finite(file.get_tensor(f'{i}.{key}'), key)
            if row['x'].shape != row['target'].shape or row['x'].shape != row['baseline'].shape:
                raise ValueError('Mismatched cached latent shapes.')
            if row['sigma'].numel() != 1 or row['sigma'].item() <= 0:
                raise ValueError('Invalid cached sigma.')
            if row['split'] not in ('train', 'validation'):
                raise ValueError('Invalid cache split.')
            rows.append(row)
    if not rows or not any(r['split'] == 'train' for r in rows) or not any(r['split'] == 'validation' for r in rows):
        raise ValueError('Cache requires separate training and validation trajectories.')
    train_keys = {(r['scene'], r['seed']) for r in rows if r['split'] == 'train'}
    val_keys = {(r['scene'], r['seed']) for r in rows if r['split'] == 'validation'}
    if train_keys & val_keys:
        raise ValueError('Training and validation trajectories overlap.')
    return rows, info


@torch.no_grad()
def euler_trajectory(predict, initial, sigmas, teacher_context, baseline_context):
    """Cache BEFORE each Euler update. Both predictions see exactly the same x_t."""
    x = initial.clone()
    rows = []
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i:i + 1]
        if sigma.item() <= 0 or sigmas[i + 1] >= sigmas[i]:
            raise ValueError('Expected a strictly decreasing positive Euler schedule ending at zero.')
        teacher = finite(predict(x, sigma, teacher_context), 'teacher output')
        baseline = finite(predict(x, sigma, baseline_context), 'baseline output')
        rows.append({'x': x.detach().cpu().clone(), 'sigma': sigma.detach().cpu().clone(),
                     'target': teacher.detach().cpu().clone(),
                     'baseline': baseline.detach().cpu().clone(), 'step': i})
        x = x + (x - teacher) / sigma.reshape(-1, *([1] * (x.ndim - 1))) * (sigmas[i + 1] - sigmas[i])
        finite(x, 'Euler latent')
    return rows, x


def match_loss(prediction, row):
    target = row['target'].to(prediction)
    baseline = row['baseline'].to(prediction)
    # Measure error relative to the actual teacher style signal, not total output energy.
    gap = (baseline.float() - target.float()).square().mean()
    energy = target.float().square().mean().clamp_min(1e-12)
    if gap <= energy * 1e-10:
        raise ValueError('Teacher and baseline denoising outputs are indistinguishable; check the profile.')
    return (prediction.float() - target.float()).square().mean() / gap


def fit(adapter_lora, rows, predict, steps, learning_rate, seed, progress=None):
    """Caller must disable inference mode and provide a differentiable predict(context)."""
    from .core import pad_context
    if not 1 <= steps <= 100 or not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.01:
        raise ValueError('Experiment allows 1–100 updates and learning_rate <= 0.01.')
    train = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    if not train or not validation:
        raise ValueError('Both cache splits are required.')
    parameters = list(adapter_lora.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0)
    generator = torch.Generator().manual_seed(seed)

    def evaluate():
        with torch.no_grad():
            return sum(match_loss(predict(r, pad_context(adapter_lora(r))), r).item()
                       for r in validation) / len(validation)

    initial = evaluate()
    if not math.isfinite(initial):
        raise ValueError('Non-finite validation loss.')
    # At zero LoRA, the live model must reproduce the cached baseline.
    if abs(initial - 1.0) > 0.05:
        raise ValueError(f'Zero-LoRA/cache parity failed ({initial:.6g}, expected ~1). Rebuild cache with the same model/runtime.')
    best_loss, best_step, best = initial, 0, None
    history = [{'step': 0, 'validation_loss': initial}]
    first_gradient = None
    started = time.perf_counter()
    order = []
    for step in range(1, steps + 1):
        if not order:
            order = torch.randperm(len(train), generator=generator).tolist()
        row = train[order.pop()]
        if progress:
            progress({'stage': 'backward_start', 'step': step, 'total': steps})
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        prediction = predict(row, pad_context(adapter_lora(row)))
        loss = finite(match_loss(prediction, row), 'loss')
        if not loss.requires_grad:
            raise RuntimeError('Denoiser detached the graph; GGUF/runtime backward is unsupported.')
        loss.backward()
        gradients = [p.grad for p in parameters if p.grad is not None]
        if len(gradients) != len(parameters):
            raise RuntimeError('Gradient did not reach both LoRA tensors.')
        norm = torch.sqrt(sum(finite(g, 'gradient').float().square().sum() for g in gradients)).item()
        if step == 1:
            if norm <= 0:
                raise RuntimeError('Zero LoRA gradient. No usable training signal.')
            first_gradient = norm
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        item = {'stage': 'updated', 'step': step, 'train_loss': loss.item(),
                'gradient_norm': norm, 'update_seconds': time.perf_counter() - tick}
        if step % 5 == 0 or step == steps:
            value = evaluate()
            if not math.isfinite(value):
                raise ValueError('Non-finite validation loss.')
            item['validation_loss'] = value
            if value < best_loss:
                best_loss, best_step = value, step
                best = (adapter_lora.down.detach().cpu().clone(), adapter_lora.up.detach().cpu().clone())
        history.append(item)
        if progress:
            progress(item)
    return best, {'initial_validation_loss': initial, 'best_validation_loss': best_loss,
                  'selected_step': best_step, 'first_gradient_norm': first_gradient,
                  'training_seconds': time.perf_counter() - started, 'history': history,
                  'objective': 'single_step_denoised_mse_divided_by_teacher_baseline_gap',
                  'visual_validation': 'not_performed'}
