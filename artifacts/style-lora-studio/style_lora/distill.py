"""TCSA absorption: rank-1 initializer, then thin-SVD / least-squares (+ ALS)."""

from __future__ import annotations

import numpy as np

from .defaults import ALS_ITERS


def _adapt_features(H: np.ndarray, in_dim: int, seed: int) -> np.ndarray:
    """Map encoder features to a layer's input width (demo / dim mismatch)."""
    d = H.shape[1]
    if d == in_dim:
        return H.astype(np.float32)
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((d, in_dim)).astype(np.float32)
    P /= np.linalg.norm(P, axis=0, keepdims=True) + 1e-8
    return (H @ P).astype(np.float32)


def rank1_mean(W: np.ndarray, H_c: np.ndarray, Delta: np.ndarray, alpha: float = 1.0):
    """Algebraically correct rank-1: ΔW = α (W μ) vᵀ with v = h̄ / ||h̄||² so vᵀ h̄ = 1.

    Do not replace the squared norm by the unsquared norm (that scales ΔW by ||h̄||).
    """
    mu = Delta.mean(axis=0)
    hbar = H_c.mean(axis=0)
    v = hbar / (float(np.dot(hbar, hbar)) + 1e-8)
    u = W @ mu
    B = (alpha * u).reshape(-1, 1).astype(np.float32)
    A = v.reshape(1, -1).astype(np.float32)
    return A, B


def rank_r_pca(W: np.ndarray, H_c: np.ndarray, Delta: np.ndarray, rank: int, alpha: float = 1.0):
    """One rank-1 absorption per uncentered PCA component, concatenated."""
    _, S, Vt = np.linalg.svd(Delta.astype(np.float64), full_matrices=False)
    r = max(1, min(int(rank), Vt.shape[0]))
    hbar = H_c.mean(axis=0).astype(np.float64)
    v_mean = hbar / (float(np.dot(hbar, hbar)) + 1e-8)
    n = max(Delta.shape[0], 1)
    Bs, As = [], []
    for k in range(r):
        mu = (Vt[k] * (S[k] / n)).astype(np.float32)
        u = W @ mu
        Bs.append(alpha * u)
        if k == 0:
            As.append(v_mean.astype(np.float32))
        else:
            vk = Vt[k].astype(np.float32)
            As.append(vk / (float(np.dot(vk, vk)) + 1e-8))
    B = np.stack(Bs, axis=1).astype(np.float32)
    A = np.stack(As, axis=0).astype(np.float32)
    return A, B


def als_refine(B: np.ndarray, A: np.ndarray, R: np.ndarray, Hc_t: np.ndarray, iters: int = ALS_ITERS):
    """A few alternating least-squares sweeps to enforce the exact BA factorization."""
    B = B.astype(np.float64)
    A = A.astype(np.float64)
    R = R.astype(np.float64)
    Hc_t = Hc_t.astype(np.float64)
    for _ in range(max(0, int(iters))):
        X = A @ Hc_t
        B = R @ np.linalg.pinv(X)
        Y = np.linalg.pinv(B) @ R
        A = Y @ np.linalg.pinv(Hc_t)
    return A, B


def svdl_absorb(
    W: np.ndarray,
    H_c: np.ndarray,
    H_s: np.ndarray,
    rank: int,
    alpha: float = 1.0,
    als_iters: int = ALS_ITERS,
):
    """Primary TCSA synthesizer: residual map thin-SVD, optional ALS.

    R = W (H_s − H_c)ᵀ,  M = R (H_cᵀ)⁺,  then rank-r SVD of M, then ALS on ||BA H_cᵀ − R||.
    """
    Delta = (H_s - H_c).T  # (in, n)
    R = W @ Delta  # (out, n)
    Hc_t = H_c.T  # (in, n)
    M = R @ np.linalg.pinv(Hc_t)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    r = max(1, min(int(rank), U.shape[1], Vt.shape[0]))
    B = U[:, :r] * S[:r]
    A = Vt[:r, :]
    if als_iters:
        A, B = als_refine(B, A, R, Hc_t, iters=als_iters)
    B = (B * float(alpha)).astype(np.float32)
    A = A.astype(np.float32)
    return A, B


def lstsq_svd(W, H_c, H_s, rank, alpha: float = 1.0):
    """Backward-compatible alias of svdl_absorb without ALS."""
    return svdl_absorb(W, H_c, H_s, rank=rank, alpha=alpha, als_iters=0)


def reconstruction_error(W, A, B, H_c, H_s) -> float:
    left = (W + B @ A) @ H_c.T
    right = W @ H_s.T
    num = np.linalg.norm(left - right)
    den = np.linalg.norm(right) + 1e-8
    return float(num / den)


def distill_layers(
    bank: dict[str, np.ndarray],
    H_c: np.ndarray,
    H_s: np.ndarray,
    rank: int = 4,
    alpha: float = 1.0,
    method: str = "svdl",
    sign: float = 1.0,
    als_iters: int = ALS_ITERS,
) -> dict:
    Delta = sign * (H_s - H_c)
    Hs = H_c + Delta
    method = {"lstsq": "svdl", "svd": "svdl", "two_stage": "svdl"}.get(method, method)
    adapters = {}
    for i, (name, W) in enumerate(bank.items()):
        if W.ndim != 2:
            continue
        out_dim, in_dim = W.shape
        Hc = _adapt_features(H_c, in_dim, seed=11 + i)
        Hst = _adapt_features(Hs, in_dim, seed=11 + i)
        dlt = Hst - Hc
        A1, B1 = rank1_mean(W, Hc, dlt, alpha=alpha)
        err1 = reconstruction_error(W, A1, B1, Hc, Hst)
        if method == "rank1":
            if int(rank) <= 1:
                A, B = A1, B1
            else:
                A, B = rank_r_pca(W, Hc, dlt, rank=rank, alpha=alpha)
        else:
            A, B = svdl_absorb(W, Hc, Hst, rank=rank, alpha=alpha, als_iters=als_iters)
        err = reconstruction_error(W, A, B, Hc, Hst)
        adapters[name] = {
            "A": A,
            "B": B,
            "in_dim": int(in_dim),
            "out_dim": int(out_dim),
            "rank": int(A.shape[0]),
            "recon_error": err,
            "rank1_recon_error": err1,
            "method": method,
        }
    return adapters


def random_matched(adapters: dict, seed: int = 0) -> dict:
    """Rank-matched noise with the same ||BA||_F (paper baseline 4)."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, spec in adapters.items():
        A = spec["A"]
        B = spec["B"]
        Ar = rng.standard_normal(A.shape).astype(np.float32)
        Br = rng.standard_normal(B.shape).astype(np.float32)
        target = float(np.linalg.norm(B @ A))
        current = float(np.linalg.norm(Br @ Ar)) + 1e-8
        Br = Br * (target / current)
        out[name] = {
            **spec,
            "A": Ar,
            "B": Br,
            "recon_error": float("nan"),
            "rank1_recon_error": float("nan"),
        }
    return out
