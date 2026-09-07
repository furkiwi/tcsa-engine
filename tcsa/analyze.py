"""Direction tests, null controls, and the soft-prompt gate (paper §4.4–4.5)."""

from __future__ import annotations

import numpy as np

from .constants import (
    EVALUATION,
    EXTRACTION,
    GATE_COSINE,
    GATE_LAMBDA1,
    GATE_SOFT,
    GATE_SPEC,
    NULLS,
    paraphrases_for,
)
from .encoder import Encoder


def pairwise_cosine(delta: np.ndarray) -> float:
    x = np.asarray(delta, dtype=np.float64)
    if x.ndim == 1:
        return 1.0
    n = x.shape[0]
    if n < 2:
        return 0.0
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-15
    g = (x / norms) @ (x / norms).T
    iu = np.triu_indices(n, k=1)
    return float(g[iu].mean()) if iu[0].size else 0.0


def lambda1_share(delta: np.ndarray) -> tuple[float, list[float]]:
    x = np.asarray(delta, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    s = np.linalg.svd(x, compute_uv=False)
    e = s**2
    tot = float(e.sum()) or 1.0
    shares = (e / tot).tolist()
    return float(shares[0]), shares


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-15
    return float(np.dot(a, b) / den)


def collect_pairs(encoder: Encoder, contents: list[str], paraphrases: list[str]) -> list:
    pairs = []
    paras = paraphrases or [""]
    for i, c in enumerate(contents):
        pairs.append(encoder.encode_pair(c, paras[i % len(paras)]))
    return pairs


def analyze(
    encoder: Encoder,
    style: str,
    paraphrases: list[str] | None = None,
    extraction: list[str] | None = None,
    evaluation: list[str] | None = None,
    on_progress: callable | None = None,
) -> dict:
    paras = paraphrases or paraphrases_for(style)
    contents = extraction or list(EXTRACTION)
    held = evaluation or list(EVALUATION)
    n = len(contents)

    pairs = []
    for i, c in enumerate(contents):
        if on_progress:
            on_progress(f"encode extraction {i + 1}/{n}")
        pairs.append(encoder.encode_pair(c, paras[i % len(paras)]))

    H_c = np.stack([p.H_c for p in pairs])
    H_s = np.stack([p.H_s for p in pairs])
    L_c = np.stack([p.layer_c for p in pairs])
    L_s = np.stack([p.layer_s for p in pairs])
    delta = H_s - H_c
    layer_delta = L_s - L_c

    cos = pairwise_cosine(delta)
    lam, spectrum = lambda1_share(delta)
    mu = delta.mean(axis=0)
    hbar = H_c.mean(axis=0)
    mu_layer = layer_delta.mean(axis=0)
    hbar_layer = L_c.mean(axis=0)

    nulls = []
    for nid, phrases in NULLS.items():
        if on_progress:
            on_progress(f"null control · {nid}")
        npairs = [encoder.encode_pair(c, phrases[i % len(phrases)]) for i, c in enumerate(contents[:12])]
        d = np.stack([p.H_s - p.H_c for p in npairs])
        nulls.append(
            {
                "id": nid,
                "phrases": phrases,
                "cosine": pairwise_cosine(d),
                "lambda1": lambda1_share(d)[0],
            }
        )
    spec = cos - max(n["cosine"] for n in nulls)

    if on_progress:
        on_progress("soft-prompt test")
    gains = []
    for c in held:
        p = encoder.encode_pair(c, paras[0])
        hsp = p.H_c + mu
        gains.append(_cos(hsp, p.H_s) - _cos(p.H_c, p.H_s))
    soft = float(np.mean(gains))

    gates = {
        "cosine": {"value": cos, "threshold": GATE_COSINE, "pass": cos >= GATE_COSINE},
        "lambda1": {"value": lam, "threshold": GATE_LAMBDA1, "pass": lam >= GATE_LAMBDA1},
        "specificity": {"value": spec, "threshold": GATE_SPEC, "pass": spec >= GATE_SPEC},
        "soft": {"value": soft, "threshold": GATE_SOFT, "pass": soft >= GATE_SOFT},
    }
    go = all(g["pass"] for g in gates.values())
    return {
        "go": go,
        "style": style,
        "paraphrases": paras,
        "encoder": encoder.name,
        "cosine": cos,
        "lambda1": lam,
        "spectrum": spectrum[:8],
        "specificity": spec,
        "soft_gain": soft,
        "nulls": nulls,
        "gates": gates,
        "mu": mu,
        "hbar": hbar,
        "H_c": H_c,
        "H_s": H_s,
        "L_c": L_c,
        "L_s": L_s,
        "mu_layer": mu_layer,
        "hbar_layer": hbar_layer,
        "n_pairs": len(pairs),
        "held_out": held,
    }
