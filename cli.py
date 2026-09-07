#!/usr/bin/env python3
"""TCSA CPU engine — extract a Krea 2 style LoRA without running the DiT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

print("TCSA CPU engine  ·  loading…", flush=True)

from tcsa.constants import DEFAULT_RANK, paraphrases_for  # noqa: E402
from tcsa.pipeline import run_job  # noqa: E402
from tcsa.raw import list_matching_keys  # noqa: E402


def _progress(msg: str) -> None:
    print(f"  · {msg}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Text-Conditioning Style Absorption for Krea 2 RAW (CPU)"
    )
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("go", help="analyze + distill + package")
    g.add_argument("--style", required=True)
    g.add_argument("--rank", type=int, default=DEFAULT_RANK)
    g.add_argument("--encoder", default="geometric", help="geometric | qwen")
    g.add_argument("--qwen", default="Qwen/Qwen3-VL-4B-Instruct")
    g.add_argument("--raw", default=None, help="path to Krea 2 RAW safetensors")
    g.add_argument("--out", default=str(ROOT / "runs"))

    i = sub.add_parser("inspect-raw", help="list txt_in / text_fusion keys in a RAW file")
    i.add_argument("--raw", required=True)

    args = p.parse_args()
    if args.cmd == "inspect-raw":
        path = Path(args.raw)
        if not path.is_file():
            print(f"missing file: {path}", flush=True)
            return 2
        found = list_matching_keys(path)
        print(json.dumps(found, indent=2), flush=True)
        if not found:
            print("No text-injection keys matched. The file may use an unexpected prefix.", flush=True)
            return 1
        return 0

    if args.cmd != "go":
        p.print_help()
        print("\nExample:\n  python cli.py go --style watercolor --encoder qwen --raw /models/krea2_raw.safetensors", flush=True)
        return 2

    print(f"style={args.style!r}  paraphrases={paraphrases_for(args.style)}", flush=True)
    print(f"encoder={args.encoder}  rank={args.rank}  raw={args.raw or '(surrogate maps)'}", flush=True)

    result = run_job(
        style=args.style,
        rank=args.rank,
        encoder=args.encoder,
        qwen=args.qwen,
        raw=args.raw,
        out_dir=args.out,
        on_progress=_progress,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, default=str), flush=True)
    s = result["summary"]
    verdict = "GO" if result["go"] else "NO-GO"
    print(
        f"\n{verdict}  cosine={s['cosine']:.3f}  λ1={s['lambda1']:.3f}  "
        f"spec={s['specificity']:.3f}  soft={s['soft_gain']:+.3f}",
        flush=True,
    )
    if result["go"]:
        print("LoRA files:", flush=True)
        for f in result["files"]:
            print(f"  {f}", flush=True)
        return 0
    print("Gates failed — no LoRA written (paper §4.4 stop rule).", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
