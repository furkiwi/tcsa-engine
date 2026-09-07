#!/usr/bin/env python3
"""Command-line entry for headless TCSA runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from style_lora.pipeline import default_spec, run_analyze, run_distill


def main():
    p = argparse.ArgumentParser(description="TCSA — CPU text-conditioning style absorption")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="encode pairs; run cosine / PCA / null / soft-prompt gates")
    a.add_argument("--config", type=Path, default=None)
    a.add_argument("--mode", choices=["demo", "real"], default=None)

    d = sub.add_parser("distill", help="write LoRA from a previous analyze job")
    d.add_argument("job_id")
    d.add_argument("--rank", type=int, default=None)
    d.add_argument("--alpha", type=float, default=None)
    d.add_argument("--method", choices=["svdl", "rank1", "lstsq", "two_stage"], default=None)
    d.add_argument("--format", dest="export_format", choices=["diffusers", "kohya"], default=None)

    g = sub.add_parser("go", help="analyze + distill in one shot")
    g.add_argument("--config", type=Path, default=None)
    g.add_argument("--mode", choices=["demo", "real"], default="demo")
    g.add_argument("--rank", type=int, default=4)
    g.add_argument("--method", choices=["svdl", "rank1", "lstsq"], default="svdl")
    g.add_argument("--out", type=Path, default=None)

    args = p.parse_args()
    if args.cmd == "analyze":
        spec = default_spec()
        if args.config:
            spec.update(json.loads(args.config.read_text(encoding="utf-8")))
        if args.mode:
            spec["mode"] = args.mode
        out = run_analyze(spec)
        s = out["stats"]
        print(json.dumps({
            "job_id": out["job_id"],
            "proceed": s["proceed"],
            "mean_cosine": s["mean_cosine"],
            "lambda1": s["lambda1"],
            "specificity": s.get("specificity"),
            "soft_prompt": s.get("soft_prompt"),
        }, indent=2, ensure_ascii=False))
    elif args.cmd == "distill":
        ov = {}
        if args.rank is not None:
            ov["rank"] = args.rank
        if args.alpha is not None:
            ov["alpha"] = args.alpha
        if args.method:
            ov["method"] = args.method
        if args.export_format:
            ov["export_format"] = args.export_format
        out = run_distill(args.job_id, ov or None)
        print(json.dumps({"lora": out["lora_path"], "layers": out["layers"]}, indent=2, ensure_ascii=False))
    elif args.cmd == "go":
        spec = default_spec()
        if args.config:
            spec.update(json.loads(args.config.read_text(encoding="utf-8")))
        spec["mode"] = args.mode
        spec["rank"] = args.rank
        spec["method"] = args.method
        analyzed = run_analyze(spec)
        distilled = run_distill(analyzed["job_id"], {"rank": args.rank, "method": args.method})
        if args.out:
            import shutil

            args.out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(distilled["lora_path"], args.out)
            print(args.out)
        else:
            print(distilled["lora_path"])
        s = analyzed["stats"]
        print(
            "proceed=", s["proceed"],
            "cosine=", round(s["mean_cosine"], 3),
            "λ1=", round(s["lambda1"], 3),
            "soft=", round(s.get("soft_prompt", {}).get("improvement", 0), 3),
        )


if __name__ == "__main__":
    main()
