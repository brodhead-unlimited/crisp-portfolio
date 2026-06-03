"""Classical baselines: 1/N, Markowitz, minimum-variance."""
from __future__ import annotations

import numpy as np

from .common import normalize_weights


def _inv(cov: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(np.asarray(cov, dtype=float))


def equal_weight(n: int) -> np.ndarray:
    """1/N portfolio."""
    return np.full(n, 1.0 / n)


def markowitz(cov, mu, normalize: str = "gross") -> np.ndarray:
    """Unconstrained mean-variance direction Sigma^{-1} mu."""
    w = _inv(cov) @ np.asarray(mu, dtype=float).ravel()
    return normalize_weights(w, mode=normalize)


def min_variance(cov, normalize: str = "long_only") -> np.ndarray:
    """Global minimum-variance portfolio Sigma^{-1} 1 / (1' Sigma^{-1} 1)."""
    inv = _inv(cov)
    ones = np.ones(cov.shape[0])
    w = inv @ ones
    return normalize_weights(w, mode=normalize)
