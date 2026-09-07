"""Longest-common-subsequence token alignment (paper §4.3)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def lcs_index_pairs(a: Sequence[int], b: Sequence[int]) -> list[tuple[int, int]]:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = dp[i - 1, j] if dp[i - 1, j] >= dp[i, j - 1] else dp[i, j - 1]
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1, j] >= dp[i, j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def pool_mean(features: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean-pool a (S, D) or (S, L, D) matrix. Optional boolean mask over S."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        return x
    if mask is None:
        return x.mean(axis=0)
    w = np.asarray(mask, dtype=np.float64).reshape(-1)
    if w.shape[0] != x.shape[0]:
        return x.mean(axis=0)
    denom = float(w.sum()) or 1.0
    return (x * w.reshape(-1, *([1] * (x.ndim - 1)))).sum(axis=0) / denom


def split_aligned(
    ids_c: Sequence[int],
    ids_s: Sequence[int],
) -> tuple[list[int], list[int], list[int]]:
    """Return (content_idx_in_c, content_idx_in_s, style_span_idx_in_s)."""
    pairs = lcs_index_pairs(list(ids_c), list(ids_s))
    c_idx = [i for i, _ in pairs]
    s_idx = [j for _, j in pairs]
    aligned_s = set(s_idx)
    span = [j for j in range(len(ids_s)) if j not in aligned_s]
    return c_idx, s_idx, span
