"""Performance metrics for a daily portfolio return series."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ann_factor(periods_per_year: int) -> float:
    return float(periods_per_year)


def annualized_return(r: pd.Series, ppy: int = 252) -> float:
    return float(r.mean() * ppy)


def annualized_vol(r: pd.Series, ppy: int = 252) -> float:
    return float(r.std(ddof=1) * np.sqrt(ppy))


def sharpe(r: pd.Series, rf: float = 0.0, ppy: int = 252) -> float:
    excess = r - rf / ppy
    sd = excess.std(ddof=1)
    return float(np.sqrt(ppy) * excess.mean() / sd) if sd > 0 else 0.0


def max_drawdown(r: pd.Series) -> float:
    curve = (1.0 + r).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())


def summary(r: pd.Series, ppy: int = 252) -> dict[str, float]:
    return {
        "ann_return": annualized_return(r, ppy),
        "ann_vol": annualized_vol(r, ppy),
        "sharpe": sharpe(r, ppy=ppy),
        "max_drawdown": max_drawdown(r),
    }
