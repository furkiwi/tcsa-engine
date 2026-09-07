"""Sanity checks for TCSA math, gates, and safetensors round-trip."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from style_lora.analyze import analyze, pairwise_cosine, soft_prompt_test, specificity
from style_lora.distill import lstsq_svd, rank1_mean, reconstruction_error, svdl_absorb
from style_lora.encoder import DemoEncoder
from style_lora.pipeline import default_spec, list_jobs, run_analyze, run_distill
from style_lora.prompts import insert_style, pair_prompts
from style_lora.safetensors_io import iter_keys, load_tensors, read_header


def test_insert_style():
    assert insert_style("a tiger walking", "watercolor") == "a watercolor tiger walking"


def test_demo_direction_is_shared():
    enc = DemoEncoder(dim=64)
    pairs = pair_prompts(
        ["a tiger in a jungle", "a castle on a hill", "a bowl of fruit", "a red motorcycle"],
        ["watercolor"],
    )
    res = enc.encode_pairs(pairs, ["watercolor"])
    stats = analyze(res, cosine_gate=0.4, pca_gate=0.3)
    assert stats["mean_cosine"] > 0.5, stats["mean_cosine"]
    assert stats["proceed"]
    assert len(stats["scatter"]) == 4


def test_unknown_style_is_weaker():
    enc = DemoEncoder(dim=64)
    known = enc.encode_pairs(
        pair_prompts(["a tiger", "a castle", "a boat", "a train"], ["watercolor"]),
        ["watercolor"],
    )
    unknown = enc.encode_pairs(
        pair_prompts(["a tiger", "a castle", "a boat", "a train"], ["glorblesh style"]),
        ["glorblesh style"],
    )
    ck = pairwise_cosine(known.Delta)
    cu = pairwise_cosine(unknown.Delta)
    n = ck.shape[0]
    mask = ~np.eye(n, dtype=bool)
    assert float(ck[mask].mean()) > float(cu[mask].mean())
    ustats = analyze(unknown, cosine_gate=0.4, pca_gate=0.35)
    assert not ustats["proceed"]


def test_rank1_recovers_additive_shift():
    rng = np.random.default_rng(0)
    inn, out, n = 32, 24, 12
    W = rng.standard_normal((out, inn)).astype(np.float32)
    H_c = rng.standard_normal((n, inn)).astype(np.float32)
    mu = rng.standard_normal(inn).astype(np.float32) * 0.4
    H_s = H_c + mu
    A, B = rank1_mean(W, H_c, H_s - H_c, alpha=1.0)
    err = reconstruction_error(W, A, B, H_c, H_s)
    hbar = H_c.mean(axis=0)
    # vᵀ h̄ = 1, so (W+BA) h̄ = W h̄ + Wμ
    left = (W + B @ A) @ hbar
    right = W @ (hbar + mu)
    assert np.linalg.norm(left - right) < 1e-4
    v = A.reshape(-1)
    assert abs(float(np.dot(v, hbar)) - 1.0) < 1e-5
    assert err < 0.35


def test_svdl_low_error():
    rng = np.random.default_rng(1)
    inn, out, n = 20, 16, 10
    W = rng.standard_normal((out, inn)).astype(np.float32)
    H_c = rng.standard_normal((n, inn)).astype(np.float32)
    mu = rng.standard_normal(inn).astype(np.float32) * 0.5
    H_s = H_c + mu
    A, B = svdl_absorb(W, H_c, H_s, rank=2, alpha=1.0, als_iters=4)
    err = reconstruction_error(W, A, B, H_c, H_s)
    assert err < 0.15, err
    A0, B0 = lstsq_svd(W, H_c, H_s, rank=2, alpha=1.0)
    err0 = reconstruction_error(W, A0, B0, H_c, H_s)
    assert err0 < 0.2


def test_soft_prompt_and_specificity():
    enc = DemoEncoder(dim=64)
    contents = ["a tiger", "a castle", "a boat", "a train", "a fox"]
    known = enc.encode_pairs(pair_prompts(contents, ["watercolor"]), ["watercolor"])
    mu = known.Delta.mean(axis=0)
    eval_pairs = pair_prompts(["a girl in a forest", "a dog on a porch"], ["watercolor"])
    ev = enc.encode_pairs(eval_pairs, ["watercolor"])
    soft = soft_prompt_test(ev.H_c, ev.H_s, mu, alpha=1.0)
    assert soft["pass"], soft
    assert soft["cos_soft"] > soft["cos_base"]

    adj = enc.encode_pairs(pair_prompts(contents, ["yellow"]), ["yellow"])
    nonce = enc.encode_pairs(pair_prompts(contents, ["xyzzorp glorblesh"]), ["xyzzorp glorblesh"])
    style_cos = analyze(known)["mean_cosine"]
    nulls = {
        "adjective": {"mean_cosine": analyze(adj)["mean_cosine"]},
        "nonce": {"mean_cosine": analyze(nonce)["mean_cosine"]},
    }
    spec = specificity(style_cos, nulls, margin=0.05)
    assert spec["pass"], spec
    assert spec["null_floor"] < style_cos


def test_end_to_end_export():
    spec = default_spec()
    spec["extraction_contents"] = spec["extraction_contents"][:6]
    spec["evaluation_contents"] = spec["evaluation_contents"][:3]
    spec["include_control"] = True
    spec["export_kohya"] = True
    spec["include_random"] = True
    spec["include_negative"] = True
    spec["include_rank1"] = True
    spec["method"] = "svdl"
    spec["rank"] = 4
    out = run_analyze(spec)
    assert out["stats"]["proceed"]
    assert out["stats"]["soft_pass"]
    assert out["stats"]["specificity_pass"]
    assert out["stats"]["control"]["proceed"] is False
    assert out["stats"]["nulls"]["adjective"]["mean_cosine"] < out["stats"]["mean_cosine"]
    dist = run_distill(out["job_id"], {"rank": 4, "method": "svdl", "export_format": "diffusers"})
    path = Path(dist["lora_path"])
    assert path.exists()
    tensors = load_tensors(path)
    assert any("lora_A.weight" in k for k in tensors)
    assert any("text_fusion.projector" in k or "txt_in.linear_1" in k for k in tensors)
    kinds = {f["kind"] for f in dist["files"]}
    formats = {f["format"] for f in dist["files"]}
    assert "main" in kinds and "negative" in kinds and "random" in kinds and "rank1_init" in kinds
    assert "kohya" in formats
    assert dist["validation"][0]["conditions"][0]["id"] == "A"
    assert dist["lora_alpha"] == 4
    meta = read_header(path).get("__metadata__", {})
    assert meta.get("ss_network_alpha") == "4"
    alpha_key = next(k for k in tensors if k.endswith(".alpha"))
    assert float(tensors[alpha_key][0]) == 4.0
    jobs = list_jobs()
    assert any(j["job_id"] == out["job_id"] for j in jobs)
    kohya = next(f for f in dist["files"] if f["format"] == "kohya")
    keys = list(iter_keys(path.parent / kohya["name"]))
    assert any("lora_down.weight" in k for k in keys)


def test_unknown_fails_full_gates():
    spec = default_spec()
    spec["style_name"] = "xyzzorp"
    spec["style_phrases"] = ["xyzzorp glorblesh", "fnorple pigment"]
    spec["extraction_contents"] = spec["extraction_contents"][:6]
    spec["evaluation_contents"] = spec["evaluation_contents"][:3]
    out = run_analyze(spec)
    assert not out["stats"]["proceed"]


if __name__ == "__main__":
    test_insert_style()
    test_demo_direction_is_shared()
    test_unknown_style_is_weaker()
    test_rank1_recovers_additive_shift()
    test_svdl_low_error()
    test_soft_prompt_and_specificity()
    test_end_to_end_export()
    test_unknown_fails_full_gates()
    print("all tests passed")
