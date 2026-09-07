"""Serialize a loadable Krea 2 LoRA in Diffusers and native/Musubi namespaces."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def write_safetensors(path: Path, tensors: dict[str, np.ndarray], metadata: dict[str, str]) -> None:
    header: dict = {"__metadata__": {k: str(v) for k, v in metadata.items()}}
    blobs: list[bytes] = []
    offset = 0
    for name, arr in tensors.items():
        raw = np.asarray(arr, dtype=np.float16).tobytes()
        header[name] = {
            "dtype": "F16",
            "shape": [int(x) for x in arr.shape],
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    payload = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 - ((len(payload) + 8) % 8)) % 8
    payload += b" " * pad
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(payload)))
        f.write(payload)
        for b in blobs:
            f.write(b)


def _slug(style: str) -> str:
    s = "".join(ch if ch.isalnum() else "-" for ch in style.lower()).strip("-")
    return s or "style"


def export_adapters(
    layers: list[dict],
    dest: Path,
    style: str,
    rank: int,
    extra_meta: dict[str, str] | None = None,
) -> list[str]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    slug = _slug(style)
    meta = {
        "tcsa_version": "2.0",
        "ss_network_dim": str(rank),
        "ss_network_alpha": str(rank),
        "lora_alpha": str(rank),
        "style": style,
        "method": "tcsa-svdl",
        **(extra_meta or {}),
    }

    diff: dict[str, np.ndarray] = {}
    native: dict[str, np.ndarray] = {}
    for L in layers:
        r = int(L["rank"])
        # PEFT: A is [r, in], B is [out, r]
        diff[f"{L['diffusers']}.lora_A.weight"] = L["A"]
        diff[f"{L['diffusers']}.lora_B.weight"] = L["B"]
        # Kohya / Musubi: down = A [r, in], up = B [out, r]
        k = L["native"].replace(".", "_")
        native[f"lora_unet_{k}.lora_down.weight"] = L["A"]
        native[f"lora_unet_{k}.lora_up.weight"] = L["B"]
        native[f"lora_unet_{k}.alpha"] = np.array(float(r), dtype=np.float16)
        # ComfyUI diffusion_model. prefix (same A/B convention)
        native[f"diffusion_model.{L['native']}.lora.down.weight"] = L["A"]
        native[f"diffusion_model.{L['native']}.lora.up.weight"] = L["B"]

    p1 = dest / f"tcsa_{slug}_r{rank}_diffusers.safetensors"
    p2 = dest / f"tcsa_{slug}_r{rank}_musubi.safetensors"
    write_safetensors(p1, diff, {**meta, "namespace": "diffusers-peft"})
    write_safetensors(p2, native, {**meta, "namespace": "musubi-comfy"})
    return [str(p1), str(p2)]
