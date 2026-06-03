"""Moment estimators: expected returns (mu) and covariance (Sigma).

These feed every allocator.  All functions take a (T x N) returns frame and
return numpy arrays aligned to ``returns.columns``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def mean_returns(returns: pd.DataFrame) -> np.ndarray:
    """Sample mean of periodic returns."""
    return returns.mean().to_numpy()


def sample_cov(returns: pd.DataFrame) -> np.ndarray:
    """Plain sample covariance."""
    return returns.cov().to_numpy()


def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance (shrinks toward scaled identity)."""
    lw = LedoitWolf().fit(returns.to_numpy())
    return lw.covariance_


def to_corr(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a covariance into (correlation, vector of std-devs)."""
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    corr = cov / denom
    return corr, std


def nearest_spd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix onto the SPD cone (clip eigenvalues)."""
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T
