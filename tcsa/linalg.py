"""Rank-1 initializer and thin-SVD / least-squares absorption (paper §4.6–4.7)."""

from __future__ import annotations

import numpy as np


def rmsnorm(x: np.ndarray, weight: np.ndarray | None = None, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    y = x / rms
    if weight is not None:
        y = y * np.asarray(weight, dtype=np.float64)
    return y


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def rank1(W: np.ndarray, mu: np.ndarray, hbar: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Algebraically correct rank-1: v = h̄ / ||h̄||² so vᵀ h̄ = 1 (Appendix B.1)."""
    W = np.asarray(W, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    hbar = np.asarray(hbar, dtype=np.float64).reshape(-1)
    n2 = float(np.dot(hbar, hbar))
    v = hbar / n2 if n2 > 1e-15 else np.zeros_like(hbar)
    u = W @ mu
    B = (alpha * u).reshape(-1, 1)
    A = v.reshape(1, -1)
    return B, A


def _thin_factor(Y: np.ndarray, X: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Low-rank BA ≈ Y X⁺ without materializing d_out × d_in."""
    # X: din × n, Y: dout × n
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    s_inv = np.where(S > 1e-8, 1.0 / S, 0.0)
    Q = Y @ (Vt.T * s_inv)  # dout × k
    Uq, Sq, Vq = np.linalg.svd(Q, full_matrices=False)
    r = int(min(rank, max(1, int((Sq > 1e-8).sum()))))
    r = min(r, Uq.shape[1], Vq.shape[0], U.shape[1])
    B = Uq[:, :r] * np.sqrt(Sq[:r])
    A = (np.sqrt(Sq[:r])[:, None] * Vq[:r, :]) @ U.T
    return B.astype(np.float64), A.astype(np.float64)


def svdl(
    W: np.ndarray,
    H_c: np.ndarray,
    H_s: np.ndarray,
    rank: int,
    als_sweeps: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Primary synthesizer: residual map R = W(Hs − Hc), rank-r BA (paper §4.7)."""
    W = np.asarray(W, dtype=np.float64)
    Hc = np.asarray(H_c, dtype=np.float64)
    Hs = np.asarray(H_s, dtype=np.float64)
    if Hc.ndim == 1:
        Hc = Hc[None, :]
        Hs = Hs[None, :]
    X = Hc.T
    R = W @ (Hs - Hc).T
    r = max(1, min(rank, X.shape[0], X.shape[1], R.shape[0]))
    B, A = _thin_factor(R, X, r)
    if als_sweeps > 0:
        B, A = als_refine(B, A, X, R, als_sweeps)
    return B, A


def als_refine(
    B: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    R: np.ndarray,
    sweeps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """A few alternating least-squares sweeps to enforce the exact BA factorization."""
    B = np.asarray(B, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    for _ in range(max(0, sweeps)):
        Z = A @ X
        B = R @ np.linalg.pinv(Z)
        A = np.linalg.pinv(B) @ R @ np.linalg.pinv(X)
    return B, A


def recon_rel(W: np.ndarray, B: np.ndarray, A: np.ndarray, H_c: np.ndarray, H_s: np.ndarray) -> float:
    Hc = np.asarray(H_c, dtype=np.float64).T
    Hs = np.asarray(H_s, dtype=np.float64).T
    target = W @ Hs
    pred = (W + B @ A) @ Hc
    den = float(np.linalg.norm(target)) or 1.0
    return float(np.linalg.norm(pred - target) / den)
