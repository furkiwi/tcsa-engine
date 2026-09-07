"""Two-stage synthesizer: rank-1 initializer then SVD/LS absorption."""

from __future__ import annotations

import numpy as np

from .constants import ALS_SWEEPS, DEFAULT_ALPHA, TARGETS
from .linalg import gelu_tanh, rank1, recon_rel, rmsnorm, svdl


def _features_for(tid: str, analysis: dict, maps: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (W, Hc, Hs, hbar, mu) feature views for a target map."""
    W = maps["W"][tid]
    if tid == "projector":
        Hc, Hs = analysis["L_c"], analysis["L_s"]
        return W, Hc, Hs, analysis["hbar_layer"], analysis["mu_layer"]
    if tid == "linear_1":
        Hc = rmsnorm(analysis["H_c"], maps.get("norm"))
        Hs = rmsnorm(analysis["H_s"], maps.get("norm"))
        mu = Hs.mean(axis=0) - Hc.mean(axis=0)
        return W, Hc, Hs, Hc.mean(axis=0), mu
    # linear_2 consumes GELU(W1 x + b1)
    W1 = maps["W"]["linear_1"]
    b1 = maps["bias"].get("linear_1", np.zeros(W1.shape[0]))
    h1c = gelu_tanh(rmsnorm(analysis["H_c"], maps.get("norm")) @ W1.T + b1)
    h1s = gelu_tanh(rmsnorm(analysis["H_s"], maps.get("norm")) @ W1.T + b1)
    mu = h1s.mean(axis=0) - h1c.mean(axis=0)
    return W, h1c, h1s, h1c.mean(axis=0), mu


def distill(analysis: dict, maps: dict, rank: int = 4, alpha: float = DEFAULT_ALPHA) -> list[dict]:
    layers = []
    for tid, dkey, nkey, din, dout in TARGETS:
        W, Hc, Hs, hbar, mu = _features_for(tid, analysis, maps)
        r_cap = min(rank, W.shape[0], W.shape[1], max(1, Hc.shape[0]))
        B, A = svdl(W, Hc, Hs, r_cap, als_sweeps=ALS_SWEEPS)
        B1, A1 = rank1(W, mu, hbar, alpha=alpha)
        layers.append(
            {
                "id": tid,
                "diffusers": dkey,
                "native": nkey,
                "rank": int(B.shape[1]),
                "in": int(W.shape[1]),
                "out": int(W.shape[0]),
                "B": B,
                "A": A,
                "B1": B1,
                "A1": A1,
                "recon": recon_rel(W, B, A, Hc, Hs),
                "recon_r1": recon_rel(W, B1, A1, Hc, Hs),
            }
        )
    return layers
