"""Read only llm_adapter weights, including GGUF quantization, on CPU."""
import hashlib
from pathlib import Path

import numpy as np
import torch

PROBE = "a red cube beside a blue sphere"


def adapter_key(name):
    for prefix in ("model.diffusion_model.llm_adapter.", "diffusion_model.llm_adapter.", "llm_adapter."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return None


def read_weights(path):
    if Path(path).suffix.lower() == ".gguf":
        import gguf
        from gguf.quants import dequantize
        reader = gguf.GGUFReader(str(path))
        for tensor in reader.tensors:
            key = adapter_key(tensor.name)
            if key is None:
                continue
            shape = tuple(int(x) for x in reversed(tensor.shape))
            original = reader.get_field("comfy.gguf.orig_shape." + tensor.name)
            if original is not None:
                shape = tuple(int(original.parts[i][0]) for i in original.data)
            values = dequantize(tensor.data, tensor.tensor_type)
            yield key, torch.from_numpy(np.array(values, dtype=np.float32, copy=True).reshape(shape))
    else:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as file:
            for name in file.keys():
                key = adapter_key(name)
                if key is not None:
                    yield key, file.get_tensor(name).float()


def load_adapter(path):
    from comfy.ldm.anima.model import LLMAdapter
    import comfy.ops

    weights = dict(read_weights(path))
    if "embed.weight" not in weights:
        raise ValueError("This checkpoint has no recognizable Anima llm_adapter weights.")
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    # Strict loading detects architecture/version differences instead of guessing.
    adapter = LLMAdapter(device="meta", dtype=torch.float32, operations=comfy.ops.manual_cast)
    adapter.load_state_dict(weights, strict=True, assign=True)
    # Rotary's non-persistent buffer was constructed on CPU by upstream.
    adapter.eval().requires_grad_(False)
    return {"module": adapter, "sha256": digest.hexdigest(), "checkpoint": Path(path).name}


def raw_encode(clip, text, allow_weighted=False):
    if clip.patcher.load_device.type != "cpu":
        raise ValueError("Set your CLIP Loader device to cpu. Image sampling can still use GPU.")
    if getattr(clip.patcher, "patches", {}):
        raise ValueError("Use an unpatched Anima CLIP for reproducible extraction.")
    result = clip.encode_from_tokens(clip.tokenize(text), return_pooled=True, return_dict=True)
    ids = result.get("t5xxl_ids")
    if ids is None or ids.ndim != 1 or not 0 < ids.numel() <= 512:
        raise ValueError("Expected stock Anima dual-tokenizer conditioning with up to 512 T5 tokens.")
    weights = result.get("t5xxl_weights", torch.ones_like(ids, dtype=torch.float32))
    if weights.shape != ids.shape or not torch.isfinite(weights).all():
        raise ValueError("T5 weights must be finite and match the token IDs.")
    if not allow_weighted and not torch.equal(weights.cpu(), torch.ones_like(weights.cpu())):
        raise ValueError("Keep scenes unweighted; put weighted artist tags in the artist field.")
    result["t5xxl_weights"] = weights
    hidden = result["cond"].detach().cpu().float()
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != 1024 or hidden.shape[1] > 512:
        raise ValueError("Expected one Anima Qwen3-0.6B prompt, at most 512 Qwen tokens.")
    return ids.cpu().long(), hidden, result


@torch.inference_mode()
def encode(adapter, clip, text, allow_weighted=False):
    ids, hidden, raw = raw_encode(clip, text, allow_weighted=allow_weighted)
    context = adapter["module"](hidden, ids[None]).detach().cpu().float()
    # Match Anima.preprocess_text_embeds: apply native T5 token weights AFTER
    # the adapter. The tokenizer handles multiple artists and escaped parentheses.
    context = context * raw["t5xxl_weights"].to(context).reshape(1, -1, 1)
    return ids.tolist(), context, raw


@torch.inference_mode()
def encoder_probe(clip):
    return raw_encode(clip, PROBE)[1][0]
