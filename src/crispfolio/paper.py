"""Live paper-trading engine for a single strategy.

Runs a *self-simulated* paper portfolio: it holds a virtual cash + shares
ledger, marks to market at each new daily close, and rebalances to the
strategy's target weights on a fixed cadence — reusing the same turnover-cost
model as the backtest, so the live equity curve is a true forward continuation
of it.

This is deliberately broker-independent: Schwab exposes no API sandbox (paper
trading lives only inside thinkorswim), and its OAuth refresh token expires
every 7 days and needs an interactive browser login, so it can't drive an
unattended scheduled job. A simulated book fed by any ``DataSource`` has none
of those constraints and, being non-sensitive, is safe to publish.

The ledger is plain JSON so a scheduled job can load it, append one step, and
commit it back. Each call to :func:`step` is idempotent per trading day: if the
latest available close is one we've already recorded, it's a no-op.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DataSource
from .strategies import crisp


# How much history to pull so the strategy always has a full lookback window.
def _fetch_start(lookback: int, pad_days: int = 120) -> int:
    """Calendar days of history to request for ``lookback`` trading days."""
    # ~252 trading days per 365 calendar; pad generously for holidays/weekends.
    return int(lookback * 365 / 252) + pad_days


@dataclass
class Ledger:
    """The full paper-portfolio state, serialised to JSON verbatim."""

    strategy: str
    tickers: list[str]
    inception: str | None = None
    initial_capital: float = 100_000.0
    lookback: int = 252
    rebalance_every: int = 21
    cost_bps: float = 5.0

    cash: float = 0.0
    shares: dict[str, float] = field(default_factory=dict)
    # weights effective right now (target from the last rebalance)
    weights: dict[str, float] = field(default_factory=dict)

    last_date: str | None = None          # last close marked to market
    last_rebalance: str | None = None      # last date we traded
    days_since_rebalance: int = 0          # trading days since last trade
    rebalance_count: int = 0

    # equity curve: parallel arrays, growth-of-$1 normalised to inception
    history_dates: list[str] = field(default_factory=list)
    history_equity: list[float] = field(default_factory=list)
    history_value: list[float] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def load(cls, path: Path) -> "Ledger | None":
        if not Path(path).exists():
            return None
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json())

    def market_value(self, prices: pd.Series) -> float:
        held = sum(self.shares.get(t, 0.0) * float(prices[t]) for t in prices.index)
        return self.cash + held


def _target_weights(strategy_name: str, window: pd.DataFrame) -> np.ndarray:
    """Compute target weights for ``window`` (currently CRISP only)."""
    if strategy_name == "crisp":
        f = crisp(gamma=0.5)
    else:
        raise ValueError(f"unsupported paper strategy: {strategy_name!r}")
    w = np.asarray(f(window), dtype=float).ravel()
    if not np.all(np.isfinite(w)):
        raise ValueError("strategy produced non-finite weights")
    return w


def _rebalance(ledger: Ledger, prices: pd.Series, window: pd.DataFrame) -> None:
    """Trade the book to target weights at ``prices``, charging turnover cost."""
    tickers = list(window.columns)
    nav = ledger.market_value(prices)

    w_new = _target_weights(ledger.strategy, window)
    w_old = np.array([ledger.weights.get(t, 0.0) for t in tickers])
    turnover = float(np.abs(w_new - w_old).sum())
    cost = nav * turnover * ledger.cost_bps / 1e4

    investable = nav - cost
    new_shares: dict[str, float] = {}
    invested = 0.0
    for t, w in zip(tickers, w_new):
        dollars = investable * float(w)
        sh = dollars / float(prices[t])
        new_shares[t] = sh
        invested += sh * float(prices[t])

    ledger.shares = new_shares
    ledger.cash = investable - invested          # residual (≈0; rounding only)
    ledger.weights = {t: float(w) for t, w in zip(tickers, w_new)}
    ledger.last_rebalance = str(prices.name)
    ledger.days_since_rebalance = 0
    ledger.rebalance_count += 1


def step(ledger: Ledger, source: DataSource, *, as_of: str | None = None) -> bool:
    """Advance the ledger by at most one trading day. Returns True if it acted.

    Pulls recent prices, and for the latest close not yet recorded: marks to
    market, rebalances if the cadence is due (or it's inception), and appends a
    point to the equity history. Idempotent if there's no new close.
    """
    # Anchor the history window to as_of when given (backfill/testing), else to
    # today — otherwise a past as_of yields start > end and no data.
    anchor = pd.Timestamp(as_of) if as_of else pd.Timestamp.today()
    start = (anchor - pd.Timedelta(days=_fetch_start(ledger.lookback))).strftime("%Y-%m-%d")
    data = source.get_prices(ledger.tickers, start=start, end=as_of)
    prices = data.prices.dropna(how="any")
    # keep only tickers we actually have, in a stable order
    cols = [t for t in ledger.tickers if t in prices.columns]
    prices = prices[cols]
    rets = data.returns("simple")[cols].dropna(how="any")

    if len(prices) == 0:
        return False
    latest = prices.index[-1]
    latest_str = str(latest.date())

    # already recorded this close → nothing to do
    if ledger.last_date == latest_str:
        return False

    if len(rets) < ledger.lookback:
        raise ValueError(
            f"only {len(rets)} return rows; need {ledger.lookback} for lookback."
        )
    window = rets.iloc[-ledger.lookback :]
    last_prices = prices.iloc[-1]

    first_run = ledger.inception is None
    if first_run:
        ledger.inception = latest_str
        ledger.cash = ledger.initial_capital
        ledger.tickers = cols

    # advance the cadence counter for a normal new day
    if not first_run:
        ledger.days_since_rebalance += 1

    due = first_run or ledger.days_since_rebalance >= ledger.rebalance_every
    if due:
        _rebalance(ledger, last_prices, window)

    nav = ledger.market_value(last_prices)
    ledger.last_date = latest_str
    ledger.history_dates.append(latest_str)
    ledger.history_value.append(round(nav, 2))
    ledger.history_equity.append(round(nav / ledger.initial_capital, 6))
    return True
