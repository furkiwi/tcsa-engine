"""TCSA gates: cosine, PCA, style-specificity vs nulls, soft-prompt."""

from __future__ import annotations

import numpy as np

from .defaults import COSINE_GATE, PCA_GATE, SOFT_PROMPT_GATE, SPECIFICITY_MARGIN
from .encoder import EncodeResult


def pairwise_cosine(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Y = X / n
    return Y @ Y.T


def mean_offdiag(C: np.ndarray) -> float:
    n = C.shape[0]
    if n < 2:
        return float(C[0, 0]) if n == 1 else 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(C[mask].mean())


def row_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    na = np.linalg.norm(A, axis=1) + 1e-8
    nb = np.linalg.norm(B, axis=1) + 1e-8
    return (A * B).sum(axis=1) / (na * nb)


def pca_ratios(X: np.ndarray, max_rank: int = 12) -> dict:
    """Uncentered SVD so a shared mean direction counts as PC1."""
    X = X.astype(np.float64)
    n, d = X.shape
    k = min(max_rank, n, d)
    if k == 0:
        return {"ratios": [], "cumulative": [], "components": np.zeros((0, d)), "singular": []}
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    ev = (S**2)[:k]
    total = float((S**2).sum() + 1e-12)
    ratios = (ev / total).tolist()
    cum = np.cumsum(ratios).tolist()
    return {
        "ratios": ratios,
        "cumulative": cum,
        "components": Vt[:k],
        "singular": S[:k].tolist(),
    }


def pca_scatter(X: np.ndarray) -> list[list[float]]:
    X = X.astype(np.float64)
    if X.shape[0] == 0 or X.shape[1] == 0:
        return []
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(2, Vt.shape[0])
    xy = X @ Vt[:k].T
    if k == 1:
        xy = np.stack([xy[:, 0], np.zeros(xy.shape[0])], axis=1)
    return xy.astype(float).tolist()


def _short_label(text: str, n: int = 28) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def analyze(enc: EncodeResult, cosine_gate: float = COSINE_GATE, pca_gate: float = PCA_GATE) -> dict:
    C = pairwise_cosine(enc.Delta)
    mean_cos = mean_offdiag(C)
    pca = pca_ratios(enc.Delta)
    lam1 = pca["ratios"][0] if pca["ratios"] else 0.0
    cosine_pass = mean_cos >= cosine_gate
    pca_pass = lam1 >= pca_gate
    mu = enc.Delta.mean(axis=0)
    delta_norms = np.linalg.norm(enc.Delta, axis=1)
    pair_rows = []
    for i, p in enumerate(enc.pairs):
        pair_rows.append(
            {
                "id": int(p.get("id", i)),
                "content": p.get("content", ""),
                "styled": p.get("styled", ""),
                "phrase": p.get("phrase", ""),
                "label": _short_label(p.get("content", f"#{i}")),
                "delta_norm": float(delta_norms[i]),
                "style_tokens": p.get("style_tokens", []),
                "shared_tokens": p.get("shared_tokens", []),
            }
        )
    span_cos = None
    if enc.Delta_span is not None and len(enc.Delta_span) > 1:
        span_cos = mean_offdiag(pairwise_cosine(enc.Delta_span))
    return {
        "n_pairs": int(enc.Delta.shape[0]),
        "dim": int(enc.dim),
        "backend": enc.backend,
        "notes": enc.notes,
        "pooling": enc.pooling,
        "mean_cosine": mean_cos,
        "mean_cosine_span": span_cos,
        "cosine_matrix": C.tolist(),
        "cosine_gate": cosine_gate,
        "cosine_pass": bool(cosine_pass),
        "pca_ratios": pca["ratios"],
        "pca_cumulative": pca["cumulative"],
        "pca_gate": pca_gate,
        "lambda1": float(lam1),
        "pca_pass": bool(pca_pass),
        "proceed": bool(cosine_pass and pca_pass),
        "delta_norm_mean": float(delta_norms.mean()) if len(delta_norms) else 0.0,
        "delta_norms": delta_norms.astype(float).tolist(),
        "mu_norm": float(np.linalg.norm(mu)),
        "scatter": pca_scatter(enc.Delta),
        "pair_rows": pair_rows,
        "pairs": enc.pairs,
        "tap_layers": list(enc.tap_layers or []),
    }


def specificity(style_cosine: float, nulls: dict, margin: float = SPECIFICITY_MARGIN) -> dict:
    """Style paraphrases must be more coherent than unrelated nulls (§4.4)."""
    floor = 0.0
    if nulls:
        floor = max(float(n.get("mean_cosine", 0.0)) for n in nulls.values())
    gap = float(style_cosine) - floor
    return {
        "style_cosine": float(style_cosine),
        "null_floor": float(floor),
        "margin": gap,
        "required_margin": margin,
        "pass": bool(gap >= margin),
    }


def soft_prompt_test(
    H_c: np.ndarray,
    H_s: np.ndarray,
    mu: np.ndarray,
    alpha: float = 1.0,
    gate: float = SOFT_PROMPT_GATE,
) -> dict:
    """Phase 1b: does h + αμ move held-out content toward E(P_{c+s})?"""
    if H_c.shape != H_s.shape or H_c.shape[1] != mu.shape[0]:
        return {
            "cos_base": 0.0,
            "cos_soft": 0.0,
            "improvement": 0.0,
            "residual_base": 0.0,
            "residual_soft": 0.0,
            "pass": False,
            "n": 0,
            "note": "dim mismatch — skipped",
        }
    H_soft = H_c + float(alpha) * mu.reshape(1, -1)
    cos_base = float(row_cosine(H_c, H_s).mean())
    cos_soft = float(row_cosine(H_soft, H_s).mean())
    residual_base = float(np.linalg.norm(H_s - H_c, axis=1).mean())
    residual_soft = float(np.linalg.norm(H_s - H_soft, axis=1).mean())
    improvement = cos_soft - cos_base
    return {
        "cos_base": cos_base,
        "cos_soft": cos_soft,
        "improvement": float(improvement),
        "residual_base": residual_base,
        "residual_soft": residual_soft,
        "gate": gate,
        "n": int(H_c.shape[0]),
        "pass": bool(improvement >= gate and cos_soft > cos_base),
    }
