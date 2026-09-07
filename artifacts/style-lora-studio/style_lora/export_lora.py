"""Serialize TCSA factors to a loadable LoRA. lora_alpha = rank so slider 1.0 = constructed ΔW."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .defaults import DIFFUSERS_TO_NATIVE
from .safetensors_io import save_file


def _kohya_name(layer: str) -> str:
    native = DIFFUSERS_TO_NATIVE.get(layer, layer)
    flat = native.replace(".", "_")
    return f"lora_unet_{flat}"


def pack(
    adapters: dict,
    format: str = "diffusers",
    alpha: float = 1.0,
    metadata: dict | None = None,
) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for layer, spec in adapters.items():
        A: np.ndarray = spec["A"]
        B: np.ndarray = spec["B"]
        r = spec["rank"]
        # Paper §4.9: lora_alpha = rank, so hosted slider 1.0 applies constructed ΔW.
        # Construction-time α is already baked into B.
        alpha_tensor = np.array([float(r)], dtype=np.float32)
        if format == "kohya":
            base = _kohya_name(layer)
            tensors[f"{base}.lora_down.weight"] = A.astype(np.float32)
            tensors[f"{base}.lora_up.weight"] = B.astype(np.float32)
            tensors[f"{base}.alpha"] = alpha_tensor
        else:
            base = f"transformer.{layer}"
            tensors[f"{base}.lora_A.weight"] = A.astype(np.float32)
            tensors[f"{base}.lora_B.weight"] = B.astype(np.float32)
            tensors[f"{base}.alpha"] = alpha_tensor
        _ = alpha
    return tensors


def export_lora(
    adapters: dict,
    out_path: str | Path,
    format: str = "diffusers",
    alpha: float = 1.0,
    metadata: dict | None = None,
) -> Path:
    tensors = pack(adapters, format=format, alpha=alpha, metadata=metadata)
    rank = next(iter(adapters.values()))["rank"]
    meta = {
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(rank),
        "ss_network_alpha": str(rank),
        "ss_base_model_name": "krea-2-raw",
        "style_lora_studio": "0.3.0",
        "tcsa": "0.3.0",
        "method": "tcsa-svdl",
    }
    if metadata:
        meta.update({k: str(v) for k, v in metadata.items()})
    path = Path(out_path)
    save_file(tensors, path, metadata=meta)
    return path
