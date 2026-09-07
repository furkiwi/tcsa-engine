#!/usr/bin/env python3
"""CPU-side checks for the paper algebra — no Qwen weights required."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcsa.align import lcs_index_pairs
from tcsa.analyze import analyze
from tcsa.constants import insert_style
from tcsa.distill import distill
from tcsa.encoder import GeometricEncoder
from tcsa.export_lora import write_safetensors
from tcsa.linalg import rank1
from tcsa.pipeline import run_job
from tcsa.raw import load_text_maps


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_insert_style() -> None:
    assert_true(insert_style("a tiger walking", "watercolor") == "a watercolor tiger walking", "prefix a")
    assert_true(insert_style("an old man", "oil painting") == "an oil painting old man", "prefix an")


def test_lcs() -> None:
    a = [1, 2, 3, 4]
    b = [1, 9, 2, 3, 4]
    pairs = lcs_index_pairs(a, b)
    assert_true([(i, a[i]) for i, _ in pairs] == [(0, 1), (1, 2), (2, 3), (3, 4)], "lcs keeps content")


def test_rank1_identity() -> None:
    rng = np.random.default_rng(0)
    W = rng.normal(size=(32, 16))
    hbar = rng.normal(size=16)
    mu = rng.normal(size=16)
    alpha = 1.0
    B, A = rank1(W, mu, hbar, alpha)
    left = (W + B @ A) @ hbar
    right = W @ (hbar + alpha * mu)
    err = float(np.linalg.norm(left - right) / (np.linalg.norm(right) + 1e-15))
    assert_true(err < 1e-8, f"rank-1 identity broken: {err}")


def test_gates() -> None:
    enc = GeometricEncoder()
    ok = analyze(enc, "watercolor")
    assert_true(ok["go"], f"watercolor should GO, gates={ok['gates']}")
    bad = analyze(enc, "xyzzorpian gloss")
    assert_true(not bad["go"], f"nonce should NO-GO, gates={bad['gates']}")


def test_pipeline_and_files(tmp: Path) -> None:
    result = run_job(style="watercolor", rank=4, encoder="geometric", out_dir=tmp)
    assert_true(result["go"], "pipeline GO")
    assert_true(len(result["files"]) == 2, "two namespaces")
    maps = load_text_maps(None)
    # rank-1 identity on projector with analysis-sized features is already covered;
    # here we just check packaged shapes.
    for path in result["files"]:
        p = Path(path)
        assert_true(p.is_file() and p.stat().st_size > 64, f"empty {p}")
        with p.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        keys = [k for k in header if k != "__metadata__"]
        assert_true(len(keys) >= 3, f"too few tensors in {p.name}: {keys}")
        meta = header["__metadata__"]
        assert_true(meta.get("lora_alpha") == "4", "alpha = rank")
        assert_true(meta.get("ss_network_dim") == "4", "dim = rank")


def test_safetensors_roundtrip(tmp: Path) -> None:
    path = tmp / "t.safetensors"
    write_safetensors(path, {"a.weight": np.ones((2, 3))}, {"k": "v"})
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    assert_true(header["a.weight"]["shape"] == [2, 3], "shape")
    assert_true(header["__metadata__"]["k"] == "v", "meta")


if __name__ == "__main__":
    tmp = Path("/tmp/tcsa-test")
    tmp.mkdir(exist_ok=True)
    test_insert_style()
    test_lcs()
    test_rank1_identity()
    test_safetensors_roundtrip(tmp)
    test_gates()
    test_pipeline_and_files(tmp)
    print("all tcsa engine tests passed")
