"""Experimental ComfyUI Anima bridge; no promise of GGUF backward compatibility."""
from contextlib import contextmanager
import hashlib
import json

import torch
from torch.utils.checkpoint import checkpoint

from .core import pad_context, transplant


def sampled_fingerprint(model):
    """Compatibility guard, NOT a full checkpoint checksum. Includes every tensor's sample."""
    digest = hashlib.sha256()
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype), str(getattr(value, 'tensor_type', '')))).encode())
        # GGMLTensor.shape describes logical dimensions, not the packed byte storage.
        raw = value.detach().as_subclass(torch.Tensor).reshape(-1)
        if raw.device.type == 'meta':
            raise ValueError('Meta/dynamic weights are unsupported in this experiment.')
        if raw.numel():
            indices = torch.linspace(0, raw.numel() - 1, min(64, raw.numel()), device=raw.device).long()
            digest.update(raw[indices].float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_model(patcher):
    model = patcher.model
    diffusion = getattr(model, 'diffusion_model', None)
    if diffusion is None or not hasattr(diffusion, 'llm_adapter') or not hasattr(diffusion, 'blocks'):
        raise ValueError('Connect the original Anima GGUF/UNet MODEL.')
    if getattr(patcher, 'patches', {}) or getattr(patcher, 'object_patches', {}):
        raise ValueError('Connect MODEL directly from the loader, without LoRA or sampling patches.')
    if any(v for v in getattr(patcher, 'model_options', {}).values()):
        raise ValueError('Model options/wrappers are unsupported. Use the original loader output.')
    if getattr(patcher, 'is_dynamic', lambda: False)():
        raise ValueError('Dynamic VRAM is unsupported for this probe. Restart with --disable-dynamic-vram if your ComfyUI supports it.')
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Expected a frozen base model.')
    return model


@contextmanager
def gradient_runtime(model):
    """Temporarily normalize inference tensors and checkpoint DiT blocks; always restore."""
    replaced = []
    forwards = []
    try:
        # Quantized integer storage is dequantized during forward, producing normal float tensors.
        # Plain inference-mode float weights/buffers need fresh storage for autograd to save them.
        for name, module in model.diffusion_model.named_modules():
            if name == 'llm_adapter' or name.startswith('llm_adapter.'):
                continue  # conditioning is computed on CPU from cached adapter features
            for table_name in ('_parameters', '_buffers'):
                table = getattr(module, table_name)
                for key, value in list(table.items()):
                    if value is None or not torch.is_inference(value):
                        continue
                    if getattr(value, 'tensor_type', None) is not None and not value.is_floating_point():
                        continue
                    # Refuse hidden LoRA patch payloads even when patcher.patches is empty.
                    if getattr(value, 'patches', []):
                        raise ValueError('Found patched GGUF weights; reload an unpatched MODEL.')
                    copy = value.detach().as_subclass(torch.Tensor).clone()
                    if table_name == '_parameters':
                        copy = torch.nn.Parameter(copy, requires_grad=False)
                    replaced.append((table, key, value))
                    table[key] = copy
        for block in model.diffusion_model.blocks:
            had_instance_forward = 'forward' in block.__dict__
            previous = block.__dict__.get('forward')
            original = block.forward

            def wrapped(*args, _original=original, **kwargs):
                # Upstream mutates block_index in a shared dict; capture it for recomputation.
                kwargs = dict(kwargs)
                if 'transformer_options' in kwargs:
                    kwargs['transformer_options'] = dict(kwargs['transformer_options'])
                if not torch.is_grad_enabled():
                    return _original(*args, **kwargs)
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)

            forwards.append((block, had_instance_forward, previous))
            block.forward = wrapped
        yield
    finally:
        for block, had, previous in reversed(forwards):
            if had:
                block.forward = previous
            else:
                del block.forward
        for table, key, original in reversed(replaced):
            table[key] = original


def model_signature(model):
    return {'model_fingerprint': sampled_fingerprint(model),
            'model_class': type(model).__name__,
            'sampling_class': type(model.model_sampling).__name__,
            'compute_dtype': str(model.get_dtype_inference())}


def load_model(patcher):
    import comfy.model_management as mm
    model = validate_model(patcher)
    mm.load_models_gpu([patcher])
    return model, patcher.load_device


@torch.no_grad()
def verify_adapter(model, adapter):
    """Catch a differently selected CPU adapter before building an invalid teacher."""
    live = model.diffusion_model.llm_adapter
    for name in ('out_proj', 'norm'):
        layer = getattr(live, name)
        weight = layer.weight
        if hasattr(layer, 'get_weight'):
            weight = layer.get_weight(weight, torch.float32)
        actual = weight.detach().as_subclass(torch.Tensor).float().cpu()
        expected = getattr(adapter['module'], name).weight.detach().float().cpu()
        if actual.shape != expected.shape or not torch.allclose(actual, expected, rtol=1e-4, atol=1e-5):
            raise ValueError('MODEL and CPU adapter weights differ at ' + name + '. Select the same checkpoint.')


def predictor(model, device, cancel=None):
    def predict(x, sigma, context):
        if cancel:
            cancel()
        # Both teacher and student compare BaseModel's denoised latent (x0), never mixed with raw velocity.
        return model.apply_model(x.to(device), sigma.to(device),
                                 c_crossattn=context.to(device), transformer_options={})
    return predict


@torch.no_grad()
def build_cache(patcher, adapter, clip, profile, scenes, width, height, seed, progress=None):
    from .backend import encode, encoder_probe
    from .denoise_core import euler_trajectory
    from comfy.samplers import calculate_sigmas
    from comfy.model_management import throw_exception_if_processing_interrupted as cancel

    prefix, probe, info = profile
    if info['adapter_sha256'] != adapter['sha256']:
        raise ValueError('Profile and CPU adapter checkpoint differ.')
    actual = encoder_probe(clip)
    if actual.shape != probe.shape or not torch.allclose(actual, probe, rtol=1e-4, atol=1e-5):
        raise ValueError('Text encoder differs from the extracted profile.')
    texts = list(dict.fromkeys(s.strip() for s in scenes.splitlines() if s.strip()))
    if not 1 <= len(texts) <= 4 or any('@' in s for s in texts):
        raise ValueError('Use 1–4 plain scene prompts, one per line, without artist tags.')
    if not (128 <= width <= 512 and 128 <= height <= 768 and width % 16 == height % 16 == 0):
        raise ValueError('Use width 128–512 and height 128–768, multiples of 16.')
    # Encode everything BEFORE loading the DiT; CPU CLIP loading can evict the denoiser.
    examples = []
    for scene in texts:
        cancel()
        captured = {}
        def capture(module, args, output):
            captured['features'] = args[0].detach().cpu().clone()
            captured['pre_norm'] = output.detach().cpu().clone()
        handle = adapter['module'].out_proj.register_forward_hook(capture)
        try:
            _, base, _ = encode(adapter, clip, scene)
        finally:
            handle.remove()
        examples.append((scene, base, captured))
    model, device = load_model(patcher)
    verify_adapter(model, adapter)
    signature = model_signature(model)
    fmt = model.latent_format
    if getattr(fmt, 'latent_channels', None) != 16 or model.diffusion_model.out_channels != 16:
        raise ValueError('This experiment currently supports 16-channel Anima image latents only.')
    sigmas = calculate_sigmas(model.model_sampling, 'sgm_uniform', 4).float().to(device)
    if len(sigmas) != 5 or sigmas[-1].item() != 0:
        raise ValueError('Unexpected four-step schedule.')
    predict = predictor(model, device, cancel)
    all_rows = []
    final = None
    for scene, base, captured in examples:
        for offset, split in ((0, 'train'), (1, 'validation')):
            cancel()
            if progress:
                progress({'stage': 'teacher_trajectory', 'scene': scene, 'seed': seed + offset, 'split': split})
            # Same CPU RNG convention as ComfyUI. Explicit singleton time dimension for Cosmos/Anima.
            noise = torch.randn((1, 16, 1, height // 8, width // 8),
                                generator=torch.Generator().manual_seed(seed + offset)).to(device)
            # ComfyUI deliberately does not shift an empty latent through process_latent_in.
            latent = torch.zeros_like(noise)
            initial = model.model_sampling.noise_scaling(sigmas[0], noise, latent, max_denoise=True)
            rows, final = euler_trajectory(predict, initial, sigmas, transplant(base, prefix, 1), pad_context(base))
            for row in rows:
                row.update(captured, base=base.detach().cpu().clone(), scene=scene, seed=seed + offset, split=split)
            all_rows.extend(rows)
    final = model.model_sampling.inverse_noise_scaling(sigmas[-1], final)
    final = model.process_latent_out(final).detach().cpu()
    signals = [((r['target'] - r['baseline']).norm() / r['target'].norm().clamp_min(1e-12)).item()
               for r in all_rows]
    metadata = dict(signature, adapter_sha256=adapter['sha256'], checkpoint=adapter['checkpoint'],
                    artist=info['artist'], scenes=texts, width=width, height=height, seed=seed,
                    sampler='euler', scheduler='sgm_uniform', steps=4, cfg=1.0,
                    sigmas=sigmas.cpu().tolist(), target_kind='BaseModel.apply_model denoised latent',
                    validation_scope='held-out seed, same prompts; not held-out scenes',
                    teacher_baseline_relative_l2_min=min(signals),
                    teacher_baseline_relative_l2_max=max(signals),
                    torch_version=torch.__version__)
    return all_rows, metadata, final


def train_cache(patcher, adapter, rows, info, steps, rank, learning_rate, seed, progress):
    from .denoise_core import fit
    from .lora_train import OutputLoRA
    from comfy.model_management import throw_exception_if_processing_interrupted as cancel

    with torch.inference_mode(False), torch.enable_grad():
        if not 1 <= rank <= 64:
            raise ValueError('Probe rank must be 1–64.')
        if info['adapter_sha256'] != adapter['sha256']:
            raise ValueError('Use the CPU adapter that created this cache.')
        model, device = load_model(patcher)
        verify_adapter(model, adapter)
        signature = model_signature(model)
        for key, value in signature.items():
            if info.get(key) != value:
                raise ValueError('Cache/model mismatch: ' + key + '. Rebuild the teacher cache.')
        # Inputs loaded during ComfyUI inference mode must become ordinary tensors.
        rows = [{k: v.detach().clone() if torch.is_tensor(v) else v for k, v in r.items()} for r in rows]
        norm = adapter['module'].norm
        student = OutputLoRA(rows[0]['features'].shape[-1], rows[0]['base'].shape[-1], rank,
                             norm.weight, norm.eps, torch.Generator().manual_seed(seed))
        for row in rows:
            if not torch.allclose(student(row), row['base'], rtol=1e-4, atol=1e-5):
                raise ValueError('CPU adapter normalization does not reproduce the cached base.')
        forward = predictor(model, device, cancel)
        def predict(row, context):
            return forward(row['x'], row['sigma'], context)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        # Offload saved tensors; block recomputation avoids retaining all dequantized weights.
        with gradient_runtime(model), torch.autograd.graph.save_on_cpu(pin_memory=False):
            best, report = fit(student, rows, predict, steps, learning_rate, seed, progress)
        report['peak_cuda_allocated_mb'] = (torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None)
        report['device'] = str(device)
        return best, report
