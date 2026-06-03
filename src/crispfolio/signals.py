"""Expected-return signals (alpha) to feed the signal-aware allocators.

Each signal maps a lookback window of returns -> a length-N vector of expected
per-period returns, on the same scale as the covariance so that mean-variance
trade-offs are sensible.  The sample mean is the naive baseline; momentum and
trend are standard cross-sectional alphas that are far less noisy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sample_mean(window: pd.DataFrame) -> np.ndarray:
    return window.mean().to_numpy()


def momentum(window: pd.DataFrame, skip: int = 21) -> np.ndarray:
    """12-1 style momentum: cumulative return over the window, excluding the
    most recent ``skip`` periods (to avoid short-term reversal).  Rescaled to a
    per-period mean so it lives on the same scale as the covariance.
    """
    if len(window) <= skip + 5:
        return sample_mean(window)
    past = window.iloc[:-skip] if skip > 0 else window
    cum = (1.0 + past).prod() - 1.0          # total return over the formation window
    per_period = cum / len(past)             # back to per-period scale
    return per_period.to_numpy()


def trend(window: pd.DataFrame, fast: int = 21, slow: int = 126) -> np.ndarray:
    """Time-series trend: per-period mean over a faster recent window, which
    leans into assets that have been rising lately.
    """
    f = min(fast, len(window))
    return window.iloc[-f:].mean().to_numpy()


def zscore_cross_section(mu: np.ndarray, target_vol: float = 0.0) -> np.ndarray:
    """Cross-sectionally standardise a raw signal (mean 0, unit dispersion)."""
    mu = np.asarray(mu, dtype=float)
    sd = mu.std()
    if sd <= 0:
        return mu
    z = (mu - mu.mean()) / sd
    return z * target_vol if target_vol > 0 else z


SIGNALS = {
    "sample_mean": sample_mean,
    "momentum": momentum,
    "trend": trend,
}
