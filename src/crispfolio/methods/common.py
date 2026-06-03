"""Shared weight post-processing."""
from __future__ import annotations

import numpy as np


def normalize_weights(w: np.ndarray, mode: str = "gross") -> np.ndarray:
    """Scale a raw weight vector to a tradable portfolio.

    mode:
      'gross'      -> sum of absolute weights == 1 (allows shorts, fully invested)
      'net'        -> weights sum to 1 (long/short net of 100%)
      'long_only'  -> clip negatives to 0, then renormalise to sum 1
      'none'       -> return as-is
    """
    w = np.asarray(w, dtype=float)
    if mode == "none":
        return w
    if mode == "long_only":
        w = np.clip(w, 0.0, None)
        s = w.sum()
        return w / s if s > 0 else np.full_like(w, 1.0 / len(w))
    if mode == "net":
        s = w.sum()
        return w / s if abs(s) > 1e-12 else w
    if mode == "gross":
        s = np.abs(w).sum()
        return w / s if s > 1e-12 else w
    raise ValueError(f"unknown normalize mode: {mode!r}")
