#!/usr/bin/env python3
"""TCSA Studio — CPU Text-Conditioning Style Absorption GUI backend."""

from __future__ import annotations

import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from style_lora import __version__
from style_lora.defaults import (
    DEFAULT_LAYERS,
    DEFAULT_METHOD,
    DEFAULT_RANK,
    EVALUATION_CONTENTS,
    EXTRACTION_CONTENTS,
    METHOD_NAME,
    STYLE_PHRASES,
)
from style_lora.pipeline import (
    default_spec,
    list_jobs,
    load_job,
    run_analyze,
    run_distill,
    RUNS,
)
from style_lora.weights import verify_keys

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

app = FastAPI(title="TCSA Studio", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Spec(BaseModel):
    style_name: str = "watercolor"
    style_phrases: list[str] = Field(default_factory=lambda: list(STYLE_PHRASES))
    extraction_contents: list[str] = Field(default_factory=lambda: list(EXTRACTION_CONTENTS))
    evaluation_contents: list[str] = Field(default_factory=lambda: list(EVALUATION_CONTENTS))
    pairing: str = "round_robin"
    mode: str = "demo"
    te_path: str = ""
    raw_path: str = ""
    te_template: str = ""
    layers: list[str] = Field(default_factory=lambda: list(DEFAULT_LAYERS))
    rank: int = DEFAULT_RANK
    alpha: float = 1.0
    method: str = DEFAULT_METHOD
    als_iters: int = 4
    export_format: str = "diffusers"
    cosine_gate: float = 0.4
    pca_gate: float = 0.35
    include_control: bool = True
    export_kohya: bool = True
    include_negative: bool = False
    include_random: bool = False
    include_rank1: bool = True


class DistillBody(BaseModel):
    job_id: str
    rank: int | None = None
    alpha: float | None = None
    method: str | None = None
    als_iters: int | None = None
    export_format: str | None = None
    layers: list[str] | None = None
    mode: str | None = None
    raw_path: str | None = None
    export_kohya: bool | None = None
    include_negative: bool | None = None
    include_random: bool | None = None
    include_rank1: bool | None = None


class VerifyBody(BaseModel):
    raw_path: str
    layers: list[str] = Field(default_factory=lambda: list(DEFAULT_LAYERS))


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "app": "tcsa-studio",
        "method": METHOD_NAME,
        "version": __version__,
        "dit": False,
    }


@app.get("/api/defaults")
def defaults():
    return default_spec()


@app.get("/api/jobs")
def api_jobs():
    return {"jobs": list_jobs()}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    try:
        return load_job(job_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/api/analyze")
def api_analyze(spec: Spec):
    try:
        return run_analyze(spec.model_dump())
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(400, str(e)) from e


@app.post("/api/distill")
def api_distill(body: DistillBody):
    override = {k: v for k, v in body.model_dump().items() if k != "job_id" and v is not None}
    try:
        return run_distill(body.job_id, override)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(400, str(e)) from e


@app.post("/api/verify-keys")
def api_verify(body: VerifyBody):
    try:
        return {"report": verify_keys(body.raw_path, body.layers)}
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/download/{job_id}/{name}")
def api_download(job_id: str, name: str):
    path = (RUNS / job_id / name).resolve()
    runs = RUNS.resolve()
    if runs not in path.parents:
        raise HTTPException(400, "invalid path")
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "file not found")
    if path.suffix not in {".safetensors", ".json", ".npz"}:
        raise HTTPException(400, "refusing to serve this file type")
    return FileResponse(path, filename=name)


app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
