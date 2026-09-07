"""TCSA job: pair → encode → gates / nulls / soft-prompt → absorb → export."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np

from .analyze import analyze, soft_prompt_test, specificity
from .defaults import (
    ALS_ITERS,
    COSINE_GATE,
    DEFAULT_LAYERS,
    DEFAULT_METHOD,
    DEFAULT_RANK,
    EVALUATION_CONTENTS,
    EXTRACTION_CONTENTS,
    LAYER_HELP,
    METHOD_FULL,
    METHOD_NAME,
    NULL_CONTROLS,
    PCA_GATE,
    SOFT_PROMPT_GATE,
    SPECIFICITY_MARGIN,
    STYLE_NAME,
    STYLE_PHRASES,
    STYLE_PRESETS,
)
from .distill import distill_layers, random_matched
from .encoder import EncodeResult, get_encoder
from .export_lora import export_lora
from .prompts import all_phrase_pairs, insert_style, pair_prompts
from .safetensors_io import iter_keys
from .weights import load_text_injection, random_layer_bank, verify_keys

RUNS = Path(__file__).resolve().parent.parent / "runs"
RUNS.mkdir(exist_ok=True)


def default_spec() -> dict:
    return {
        "style_name": STYLE_NAME,
        "style_phrases": list(STYLE_PHRASES),
        "extraction_contents": list(EXTRACTION_CONTENTS),
        "evaluation_contents": list(EVALUATION_CONTENTS),
        "pairing": "round_robin",
        "mode": "demo",
        "te_path": "",
        "raw_path": "",
        "te_template": "",
        "layers": list(DEFAULT_LAYERS),
        "rank": DEFAULT_RANK,
        "alpha": 1.0,
        "method": DEFAULT_METHOD,
        "als_iters": ALS_ITERS,
        "export_format": "diffusers",
        "cosine_gate": COSINE_GATE,
        "pca_gate": PCA_GATE,
        "include_control": True,
        "export_kohya": True,
        "include_negative": False,
        "include_random": False,
        "include_rank1": True,
        "presets": STYLE_PRESETS,
        "layer_help": LAYER_HELP,
        "null_controls": NULL_CONTROLS,
        "method_name": METHOD_NAME,
        "method_full": METHOD_FULL,
    }


def build_pairs(spec: dict) -> list[dict]:
    contents = [c.strip() for c in spec["extraction_contents"] if c.strip()]
    phrases = [p.strip() for p in spec["style_phrases"] if p.strip()]
    if spec.get("pairing") == "cartesian":
        return all_phrase_pairs(contents, phrases)
    return pair_prompts(contents, phrases)


def _slim_stats(stats: dict) -> dict:
    keep = (
        "mean_cosine",
        "lambda1",
        "proceed",
        "cosine_pass",
        "pca_pass",
        "delta_norm_mean",
        "n_pairs",
        "dim",
    )
    return {k: stats.get(k) for k in keep}


def run_analyze(spec: dict) -> dict:
    pairs = build_pairs(spec)
    if len(pairs) < 3:
        raise ValueError("至少需要 3 組內容 prompt 才能做跨概念方向檢驗")
    enc = get_encoder(
        spec.get("mode", "demo"),
        spec.get("te_path") or None,
        instruction_template=spec.get("te_template") or "",
    )
    encoded: EncodeResult = enc.encode_pairs(pairs, spec["style_phrases"])
    cg = float(spec.get("cosine_gate", COSINE_GATE))
    pg = float(spec.get("pca_gate", PCA_GATE))
    stats = analyze(encoded, cosine_gate=cg, pca_gate=pg)

    contents = [p["content"] for p in pairs]
    nulls = {}
    if spec.get("include_control", True):
        for key, ctrl in NULL_CONTROLS.items():
            npairs = pair_prompts(contents, ctrl["phrases"])
            nenc = enc.encode_pairs(npairs, ctrl["phrases"])
            nstats = analyze(nenc, cosine_gate=cg, pca_gate=pg)
            nulls[key] = {
                "id": ctrl["id"],
                "label": ctrl["label"],
                "phrases": ctrl["phrases"],
                "note": ctrl["note"],
                **_slim_stats(nstats),
            }
    spec_gate = specificity(stats["mean_cosine"], nulls, margin=SPECIFICITY_MARGIN)
    stats["nulls"] = nulls
    stats["specificity"] = spec_gate
    stats["specificity_pass"] = spec_gate["pass"]
    # Backward-compatible single control card (nonce).
    if "nonce" in nulls:
        stats["control"] = {
            "phrase": " / ".join(nulls["nonce"]["phrases"]),
            **{k: nulls["nonce"][k] for k in ("mean_cosine", "lambda1", "proceed", "cosine_pass", "pca_pass", "delta_norm_mean")},
        }

    evals = [c.strip() for c in (spec.get("evaluation_contents") or []) if c.strip()]
    phrases = [p.strip() for p in spec.get("style_phrases", []) if p.strip()]
    mu = encoded.Delta.mean(axis=0)
    if evals and phrases:
        eval_pairs = pair_prompts(evals, [phrases[0]])
        eval_enc = enc.encode_pairs(eval_pairs, phrases)
        soft = soft_prompt_test(
            eval_enc.H_c,
            eval_enc.H_s,
            mu,
            alpha=float(spec.get("alpha", 1.0)),
            gate=SOFT_PROMPT_GATE,
        )
        H_eval_c, H_eval_s = eval_enc.H_c, eval_enc.H_s
    else:
        soft = {
            "cos_base": 0.0,
            "cos_soft": 0.0,
            "improvement": 0.0,
            "pass": False,
            "n": 0,
            "note": "no held-out prompts",
        }
        H_eval_c = np.zeros((0, encoded.dim), dtype=np.float32)
        H_eval_s = H_eval_c
    stats["soft_prompt"] = soft
    stats["soft_pass"] = bool(soft.get("pass"))
    stats["proceed"] = bool(
        stats["cosine_pass"] and stats["pca_pass"] and spec_gate["pass"] and stats["soft_pass"]
    )
    stats["gates"] = {
        "cosine": stats["cosine_pass"],
        "pca": stats["pca_pass"],
        "specificity": spec_gate["pass"],
        "soft_prompt": stats["soft_pass"],
    }

    job_id = uuid.uuid4().hex[:12]
    job_dir = RUNS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    payload_npz = {
        "H_c": encoded.H_c,
        "H_s": encoded.H_s,
        "Delta": encoded.Delta,
        "mu": mu.astype(np.float32),
        "H_eval_c": H_eval_c,
        "H_eval_s": H_eval_s,
    }
    if encoded.Delta_span is not None:
        payload_npz["Delta_span"] = encoded.Delta_span
    np.savez_compressed(job_dir / "features.npz", **payload_npz)

    slim_spec = {k: v for k, v in spec.items() if k not in {"presets", "layer_help", "null_controls"}}
    payload = {
        "job_id": job_id,
        "spec": slim_spec,
        "stats": stats,
        "backend": encoded.backend,
        "dim": encoded.dim,
        "created": time.time(),
        "n_pairs": stats["n_pairs"],
        "method_name": METHOD_NAME,
        "preview_pairs": [
            {"content": p["content"], "styled": p["styled"], "phrase": p["phrase"]}
            for p in pairs[:8]
        ],
    }
    (job_dir / "analyze.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _layer_bank(spec: dict, H_c: np.ndarray):
    layers = spec.get("layers") or list(DEFAULT_LAYERS)
    if spec.get("mode") == "real" and spec.get("raw_path"):
        bank, report = load_text_injection(spec["raw_path"], layers, return_report=True)
        return bank, "raw-checkpoint", report
    from .defaults import DEMO_LAYERS

    bank = random_layer_bank(seed=7)
    if "txt_in.linear_1" in bank and H_c.shape[1] != bank["txt_in.linear_1"].shape[1]:
        rng = np.random.default_rng(7)
        out, _ = DEMO_LAYERS["txt_in.linear_1"]
        inn = H_c.shape[1]
        W = rng.standard_normal((out, inn)).astype(np.float32)
        q, _ = np.linalg.qr(W.T)
        bank["txt_in.linear_1"] = (q.T[:out] / (inn**0.5)).astype(np.float32)
    bank = {k: v for k, v in bank.items() if k in layers}
    report = [
        {
            "requested": name,
            "native_alias": "",
            "found": name,
            "verified": False,
            "shape": list(W.shape),
            "note": "demo-random-W — not a Krea checkpoint",
        }
        for name, W in bank.items()
    ]
    return bank, "demo-random-W", report


def _layer_summary(adapters: dict) -> dict:
    def _num(x):
        return None if x != x else x

    return {
        k: {
            "rank": v["rank"],
            "in_dim": v["in_dim"],
            "out_dim": v["out_dim"],
            "recon_error": _num(v["recon_error"]),
            "rank1_recon_error": _num(v.get("rank1_recon_error", float("nan"))),
            "A": list(v["A"].shape),
            "B": list(v["B"].shape),
        }
        for k, v in adapters.items()
    }


def run_distill(job_id: str, spec_override: dict | None = None) -> dict:
    job_dir = RUNS / job_id
    meta = json.loads((job_dir / "analyze.json").read_text(encoding="utf-8"))
    spec = meta["spec"]
    if spec_override:
        spec = {**spec, **spec_override}
    feats = np.load(job_dir / "features.npz")
    H_c, H_s = feats["H_c"], feats["H_s"]
    bank, weight_src, key_report = _layer_bank(spec, H_c)
    if not bank:
        raise ValueError("沒有選到任何目標層")
    rank = int(spec.get("rank", DEFAULT_RANK))
    alpha = float(spec.get("alpha", 1.0))
    method = spec.get("method", DEFAULT_METHOD)
    als_iters = int(spec.get("als_iters", ALS_ITERS))
    adapters = distill_layers(
        bank,
        H_c,
        H_s,
        rank=rank,
        alpha=alpha,
        method=method,
        sign=1.0,
        als_iters=als_iters,
    )
    style = spec.get("style_name", "style").replace(" ", "_")
    fmt = spec.get("export_format", "diffusers")
    files = []

    def _write(adp, fmt_name: str, tag: str, extra_meta: dict | None = None):
        name = f"krea2_{style}_tcsa_r{rank}_{tag}_{fmt_name}.safetensors"
        path = job_dir / name
        export_lora(
            adp,
            path,
            format=fmt_name,
            alpha=alpha,
            metadata={
                "style_name": spec.get("style_name", ""),
                "phrases": " | ".join(spec.get("style_phrases", [])),
                "mode": spec.get("mode", "demo"),
                "method": method,
                "weight_src": weight_src,
                "tag": tag,
                "tcsa": "0.3.0",
                **(extra_meta or {}),
            },
        )
        keys = list(iter_keys(path))
        files.append({"name": name, "kind": tag, "format": fmt_name, "keys": keys})
        return name

    primary = _write(adapters, fmt, "main")
    if spec.get("export_kohya", True) and fmt != "kohya":
        _write(adapters, "kohya", "main")
    if spec.get("include_rank1", True) and method != "rank1":
        init = distill_layers(
            bank, H_c, H_s, rank=1, alpha=alpha, method="rank1", sign=1.0, als_iters=0
        )
        _write(init, fmt, "rank1_init", {"stage": "initializer"})
    if spec.get("include_negative"):
        neg = distill_layers(
            bank, H_c, H_s, rank=rank, alpha=alpha, method=method, sign=-1.0, als_iters=als_iters
        )
        _write(neg, fmt, "negative", {"sign": "-1"})
    if spec.get("include_random"):
        rnd = random_matched(adapters, seed=0)
        _write(rnd, fmt, "random", {"baseline": "random-matched"})

    formula = (
        "ΔW = α (Wμ) vᵀ,  v = h̄ / ||h̄||²"
        if method == "rank1"
        else "R = W(H_s−H_c)ᵀ;  BA ≈ SVD_r(R (H_cᵀ)⁺) + ALS"
    )
    summary = {
        "job_id": job_id,
        "lora_path": str(job_dir / primary),
        "lora_name": primary,
        "weight_src": weight_src,
        "method": method,
        "method_name": METHOD_NAME,
        "rank": rank,
        "alpha": alpha,
        "lora_alpha": rank,
        "layers": _layer_summary(adapters),
        "files": files,
        "validation": validation_cards(spec),
        "proceed": meta["stats"]["proceed"],
        "formula": formula,
        "key_report": key_report,
    }
    (job_dir / "distill.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def validation_cards(spec: dict) -> list[dict]:
    evals = spec.get("evaluation_contents") or EVALUATION_CONTENTS
    phrase = (spec.get("style_phrases") or ["style"])[0]
    groups = []
    for i, prompt in enumerate(evals):
        styled = insert_style(prompt, phrase)
        groups.append(
            {
                "group": i,
                "content": prompt,
                "conditions": [
                    {"id": "A", "title": "Baseline", "prompt": prompt, "lora": False},
                    {"id": "B", "title": "Explicit style", "prompt": styled, "lora": False},
                    {"id": "C", "title": "Proposed LoRA", "prompt": prompt, "lora": True},
                    {"id": "D", "title": "Combined", "prompt": styled, "lora": True},
                ],
            }
        )
    return groups


def list_jobs(limit: int = 24) -> list[dict]:
    rows = []
    if not RUNS.exists():
        return rows
    dirs = sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        meta_path = d / "analyze.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        distill_path = d / "distill.json"
        distill = None
        if distill_path.exists():
            try:
                distill = json.loads(distill_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                distill = None
        safes = sorted(p.name for p in d.glob("*.safetensors"))
        rows.append(
            {
                "job_id": d.name,
                "style_name": meta.get("spec", {}).get("style_name", ""),
                "mode": meta.get("spec", {}).get("mode", ""),
                "proceed": meta.get("stats", {}).get("proceed"),
                "mean_cosine": meta.get("stats", {}).get("mean_cosine"),
                "lambda1": meta.get("stats", {}).get("lambda1"),
                "created": meta.get("created"),
                "has_lora": bool(safes),
                "lora_name": (distill or {}).get("lora_name"),
                "files": safes,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def load_job(job_id: str) -> dict:
    job_dir = RUNS / job_id
    analyze_path = job_dir / "analyze.json"
    if not analyze_path.exists():
        raise FileNotFoundError(job_id)
    payload = json.loads(analyze_path.read_text(encoding="utf-8"))
    distill_path = job_dir / "distill.json"
    if distill_path.exists():
        payload["distill"] = json.loads(distill_path.read_text(encoding="utf-8"))
    payload["files"] = sorted(p.name for p in job_dir.glob("*") if p.is_file())
    return payload
