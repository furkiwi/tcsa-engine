"""End-to-end TCSA job: encode → gates → absorb → package."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .analyze import analyze
from .constants import DEFAULT_RANK, paraphrases_for
from .distill import distill
from .encoder import make_encoder
from .export_lora import export_adapters
from .raw import load_text_maps


def run_job(
    style: str,
    rank: int = DEFAULT_RANK,
    encoder: str = "geometric",
    qwen: str = "Qwen/Qwen3-VL-4B-Instruct",
    raw: str | Path | None = None,
    out_dir: str | Path = "runs",
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    enc = make_encoder(encoder, qwen_id=qwen)
    if on_progress:
        on_progress(f"encoder · {enc.name}")
    paras = paraphrases_for(style)
    analysis = analyze(enc, style, paras, on_progress=on_progress)
    summary = {
        "style": style,
        "paraphrases": paras,
        "encoder": enc.name,
        "go": analysis["go"],
        "cosine": analysis["cosine"],
        "lambda1": analysis["lambda1"],
        "spectrum": analysis["spectrum"],
        "specificity": analysis["specificity"],
        "soft_gain": analysis["soft_gain"],
        "nulls": analysis["nulls"],
        "gates": {k: {"value": v["value"], "pass": v["pass"], "threshold": v["threshold"]} for k, v in analysis["gates"].items()},
        "n_pairs": analysis["n_pairs"],
        "held_out": analysis["held_out"],
        "rank": rank,
    }
    result: dict = {"go": analysis["go"], "summary": summary, "files": [], "out": None, "layers": []}
    if not analysis["go"]:
        if on_progress:
            on_progress("NO-GO · gates failed, no LoRA written")
        return result

    if on_progress:
        on_progress("load text-injection maps")
    maps = load_text_maps(Path(raw) if raw else None)
    summary["weights"] = maps["source"]
    summary["resolved_keys"] = maps.get("resolved", {})

    if on_progress:
        on_progress("rank-1 init + SVD/LS absorption")
    layers = distill(analysis, maps, rank=rank)
    summary["mean_recon"] = sum(L["recon"] for L in layers) / len(layers)
    summary["mean_recon_r1"] = sum(L["recon_r1"] for L in layers) / len(layers)
    summary["targets"] = [
        {
            "id": L["id"],
            "diffusers": L["diffusers"],
            "native": L["native"],
            "rank": L["rank"],
            "shape": [L["out"], L["in"]],
            "recon": L["recon"],
            "recon_r1": L["recon_r1"],
        }
        for L in layers
    ]

    dest = Path(out_dir) / "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in style.lower())
    files = export_adapters(
        layers,
        dest,
        style,
        rank,
        extra_meta={
            "tcsa_encoder": enc.name,
            "tcsa_weights": maps["source"],
            "tcsa_cosine": f"{analysis['cosine']:.4f}",
            "tcsa_lambda1": f"{analysis['lambda1']:.4f}",
        },
    )
    result["files"] = files
    result["out"] = str(dest)
    result["layers"] = summary["targets"]
    result["summary"] = summary
    if on_progress:
        on_progress("packaged")
    return result
