"""Lazy-load only the Krea 2 text-injection matrices. Aliases are a checklist."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .defaults import DEMO_LAYERS, DIFFUSERS_TO_NATIVE, NATIVE_TO_DIFFUSERS
from .safetensors_io import iter_keys, load_tensors, read_header


def _norm(key: str) -> str:
    k = key.replace("model.", "").replace("diffusion_model.", "").replace("transformer.", "")
    k = k.replace("lora_unet_", "").replace("lora_transformer_", "")
    return k


def match_layer_key(available: list[str], layer: str) -> str | None:
    aliases = {layer, DIFFUSERS_TO_NATIVE.get(layer, ""), NATIVE_TO_DIFFUSERS.get(layer, "")}
    aliases.discard("")
    for key in available:
        nk = _norm(key)
        for a in aliases:
            if nk == a or nk.endswith("." + a) or nk.endswith(a):
                if any(x in key for x in (".lora_", "lora_A", "lora_B")):
                    continue
                return key
    layer_tail = layer.split(".")[-1]
    for key in available:
        if key.endswith(layer) or key.endswith(DIFFUSERS_TO_NATIVE.get(layer, "___")):
            return key
        if layer_tail in key and "txt" in key.lower():
            return key
    return None


def random_layer_bank(seed: int = 7, dim_in: int | None = None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    bank = {}
    for name, (out, inn) in DEMO_LAYERS.items():
        inn = dim_in or inn
        if "projector" in name:
            out, inn = DEMO_LAYERS[name]
        W = rng.standard_normal((out, inn)).astype(np.float32)
        q, _ = np.linalg.qr(W.T)
        W = q.T[:out].astype(np.float32) * (1.0 / np.sqrt(inn))
        bank[name] = W
    return bank


def verify_keys(raw_path: str, layers: list[str]) -> list[dict]:
    """On-disk checklist: aliases are hypotheses until a key is actually found."""
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 RAW checkpoint：{raw_path}")
    header = read_header(path)
    keys = [k for k in header if k != "__metadata__"]
    report = []
    for layer in layers:
        hit = match_layer_key(keys, layer)
        shape = header[hit]["shape"] if hit else None
        report.append(
            {
                "requested": layer,
                "native_alias": DIFFUSERS_TO_NATIVE.get(layer, ""),
                "found": hit,
                "verified": hit is not None,
                "shape": list(shape) if shape is not None else None,
                "note": "verified on disk" if hit else "NOT FOUND — do not serialize this target",
            }
        )
    return report


def load_text_injection(
    raw_path: str,
    layers: list[str],
    return_report: bool = False,
):
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 RAW checkpoint：{raw_path}")
    keys = list(iter_keys(path))
    found = {}
    missing = []
    report = verify_keys(raw_path, layers)
    for layer, row in zip(layers, report):
        if not row["verified"]:
            missing.append(layer)
            continue
        hit = row["found"]
        found[layer] = load_tensors(path, [hit])[hit].astype(np.float32)
    if missing:
        sample = "\n  ".join(keys[:12])
        raise KeyError(
            "checkpoint 裡找不到這些文本注入層（鍵名必須對盤驗證）：\n  "
            + "\n  ".join(missing)
            + "\n前幾個鍵名：\n  "
            + sample
        )
    if return_report:
        return found, report
    return found
