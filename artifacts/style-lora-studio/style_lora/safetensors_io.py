"""Minimal safetensors reader/writer (numpy only, no torch)."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterator

import numpy as np

_DTYPE = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,  # stored raw; caller may reinterpret
    "I64": np.int64,
    "I32": np.int32,
    "U8": np.uint8,
    "BOOL": np.bool_,
}

_NP = {
    np.dtype("float64"): "F64",
    np.dtype("float32"): "F32",
    np.dtype("float16"): "F16",
    np.dtype("int64"): "I64",
    np.dtype("int32"): "I32",
    np.dtype("uint8"): "U8",
    np.dtype("bool"): "BOOL",
}


def read_header(path: str | Path) -> dict:
    path = Path(path)
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header


def iter_keys(path: str | Path) -> Iterator[str]:
    header = read_header(path)
    for k in header:
        if k != "__metadata__":
            yield k


def load_tensors(path: str | Path, keys: list[str] | None = None) -> dict[str, np.ndarray]:
    """Load selected tensors without pulling the rest of a large file into RAM."""
    path = Path(path)
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data_start = 8 + n
        want = keys if keys is not None else [k for k in header if k != "__metadata__"]
        out = {}
        for key in want:
            info = header[key]
            start, end = info["data_offsets"]
            f.seek(data_start + start)
            raw = f.read(end - start)
            dt = _DTYPE[info["dtype"]]
            arr = np.frombuffer(raw, dtype=dt).reshape(info["shape"]).copy()
            out[key] = arr
    return out


def save_file(tensors: dict[str, np.ndarray], path: str | Path, metadata: dict | None = None) -> None:
    header: dict = {}
    blobs: list[bytes] = []
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        tag = _NP.get(arr.dtype)
        if tag is None:
            arr = arr.astype(np.float32)
            tag = "F32"
        blob = arr.tobytes()
        header[name] = {
            "dtype": tag,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(raw_header) % 8)) % 8
    raw_header = raw_header + b" " * pad
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(raw_header)))
        f.write(raw_header)
        for blob in blobs:
            f.write(blob)
