"""Lazy named-key reader for Krea 2 RAW safetensors. Never opens DiT blocks."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .constants import N_LAYERS, PUBLISHED_PROJECTOR, RAW_KEY_ALIASES, TARGETS

DTYPE_MAP = {
    "F16": np.float16,
    "BF16": np.float16,  # stored bits; we upcast via float16 view then float64
    "F32": np.float32,
    "F64": np.float64,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
    "I8": np.int8,
    "BOOL": np.uint8,
}


def _bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return np.frombuffer(u16.astype(np.uint32) << 16, dtype=np.float32)


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def list_matching_keys(path: Path) -> dict[str, list[str]]:
    header, _ = read_header(path)
    names = [k for k in header if k != "__metadata__"]
    found: dict[str, list[str]] = {}
    for alias, options in RAW_KEY_ALIASES.items():
        hits = [k for k in names if k in options or any(k.endswith(opt) for opt in options)]
        if hits:
            found[alias] = hits
    return found


def _pick(header: dict, options: tuple[str, ...]) -> str | None:
    keys = [k for k in header if k != "__metadata__"]
    for opt in options:
        if opt in header:
            return opt
        for k in keys:
            if k.endswith(opt) or k.endswith("." + opt):
                return k
    return None


def _load_tensor(f: BinaryIO, data_start: int, info: dict) -> np.ndarray:
    dtype = info["dtype"]
    shape = info["shape"]
    start, end = info["data_offsets"]
    f.seek(data_start + start)
    raw = f.read(end - start)
    if dtype == "BF16":
        arr = _bf16_to_f32(raw)
    else:
        np_dtype = DTYPE_MAP.get(dtype)
        if np_dtype is None:
            raise ValueError(f"unsupported safetensors dtype {dtype}")
        arr = np.frombuffer(raw, dtype=np_dtype)
    return np.asarray(arr, dtype=np.float64).reshape(shape)


def load_text_maps(path: Path | None) -> dict:
    """Return W matrices for projector / linear_1 / linear_2.

    If `path` is None, projector uses the published 1×12 mix and the txtmlp
    maps are orthonormal surrogates tagged as such. DiT blocks are never read.
    """
    out: dict = {"source": "surrogate", "path": None, "resolved": {}, "W": {}, "bias": {}, "norm": None}
    if path is None or not Path(path).is_file():
        out["W"]["projector"] = np.asarray(PUBLISHED_PROJECTOR, dtype=np.float64).reshape(1, N_LAYERS)
        rng = np.random.default_rng(20260907)
        # Orthonormal-ish surrogates so rank-1 identity tests still hold.
        W1 = rng.normal(0, 1 / np.sqrt(2560), size=(6144, 2560))
        W2 = rng.normal(0, 1 / np.sqrt(6144), size=(6144, 6144))
        out["W"]["linear_1"] = W1
        out["W"]["linear_2"] = W2
        out["bias"]["linear_1"] = np.zeros(6144)
        out["bias"]["linear_2"] = np.zeros(6144)
        out["norm"] = np.ones(2560)
        return out

    path = Path(path)
    header, data_start = read_header(path)
    out["source"] = "krea2-raw"
    out["path"] = str(path)
    with path.open("rb") as f:
        for alias, options in RAW_KEY_ALIASES.items():
            key = _pick(header, options)
            if key is None:
                continue
            tensor = _load_tensor(f, data_start, header[key])
            out["resolved"][alias] = key
            if alias == "txt_norm":
                out["norm"] = tensor.reshape(-1)
            elif alias.endswith("_bias"):
                out["bias"][alias.replace("_bias", "")] = tensor.reshape(-1)
            else:
                # Linear weights are stored [out, in]
                out["W"][alias] = tensor

    if "projector" not in out["W"]:
        out["W"]["projector"] = np.asarray(PUBLISHED_PROJECTOR, dtype=np.float64).reshape(1, N_LAYERS)
        out["source"] = "krea2-raw+published-projector"
    if "linear_1" not in out["W"] or "linear_2" not in out["W"]:
        raise RuntimeError(
            "RAW file did not contain txtmlp.1 / txtmlp.3 (or Diffusers aliases). "
            f"Resolved keys: {out['resolved']}. Open the checkpoint once with "
            "`python cli.py inspect-raw --raw PATH` and confirm Appendix A names."
        )
    out.setdefault("bias", {})
    out["bias"].setdefault("linear_1", np.zeros(out["W"]["linear_1"].shape[0]))
    out["bias"].setdefault("linear_2", np.zeros(out["W"]["linear_2"].shape[0]))
    if out["norm"] is None:
        out["norm"] = np.ones(out["W"]["linear_1"].shape[1])
    return out


def describe_targets() -> list[dict]:
    return [
        {
            "id": tid,
            "diffusers": dkey,
            "native": nkey,
            "in": din,
            "out": dout,
        }
        for tid, dkey, nkey, din, dout in TARGETS
    ]
