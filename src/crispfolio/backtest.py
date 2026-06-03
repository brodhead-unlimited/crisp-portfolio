"""Walk-forward backtest engine.

A *strategy* is a callable ``f(returns_window: DataFrame) -> np.ndarray`` that
maps a lookback window of asset returns to a weight vector aligned with the
window's columns.  The engine re-estimates and rebalances on a fixed cadence,
holds the weights between rebalances, and accounts for turnover costs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import metrics

Strategy = Callable[[pd.DataFrame], np.ndarray]


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series           # net portfolio returns, per period
    weights: pd.DataFrame        # weights effective at each rebalance
    turnover: pd.Series          # gross turnover at each rebalance
    stats: dict = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()


def run_backtest(
    returns: pd.DataFrame,
    strategy: Strategy,
    name: str = "strategy",
    lookback: int = 252,
    rebalance_every: int = 21,
    cost_bps: float = 0.0,
    ppy: int = 252,
) -> BacktestResult:
    """Roll ``strategy`` through ``returns`` and measure realised performance.

    lookback        : periods of history fed to the strategy at each rebalance
    rebalance_every : periods between rebalances (21 ~= monthly on daily data)
    cost_bps        : per-side transaction cost in basis points of turnover
    """
    returns = returns.dropna(how="any")
    dates = returns.index
    n_assets = returns.shape[1]

    w_current = np.zeros(n_assets)
    port_ret = pd.Series(0.0, index=dates, dtype=float)
    weight_log: dict[pd.Timestamp, np.ndarray] = {}
    turnover_log: dict[pd.Timestamp, float] = {}

    for t in range(lookback, len(dates)):
        # rebalance on cadence (and on the first eligible day)
        if (t - lookback) % rebalance_every == 0:
            window = returns.iloc[t - lookback : t]
            w_new = np.asarray(strategy(window), dtype=float).ravel()
            if w_new.shape[0] != n_assets or not np.all(np.isfinite(w_new)):
                w_new = w_current
            turnover = np.abs(w_new - w_current).sum()
            cost = turnover * cost_bps / 1e4
            turnover_log[dates[t]] = turnover
            weight_log[dates[t]] = w_new
            w_current = w_new
        else:
            cost = 0.0

        # realised return for day t using weights held coming into the day
        port_ret.iloc[t] = float(w_current @ returns.iloc[t].to_numpy()) - cost

    port_ret = port_ret.iloc[lookback:]
    result = BacktestResult(
        name=name,
        returns=port_ret,
        weights=pd.DataFrame.from_dict(weight_log, orient="index", columns=returns.columns),
        turnover=pd.Series(turnover_log),
        stats=metrics.summary(port_ret, ppy=ppy),
    )
    result.stats["avg_turnover"] = float(result.turnover.mean()) if len(result.turnover) else 0.0
    return result
