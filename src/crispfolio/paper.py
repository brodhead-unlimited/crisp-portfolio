"""Live paper-trading engine for a single strategy.

Runs a forward paper portfolio: at each new daily close it marks the book to
market and, on a fixed cadence, rebalances to the strategy's target weights by
generating and routing **discrete whole-share orders** through a
:class:`~crispfolio.broker.base.Broker`. With a ``PaperBroker`` the fills are
simulated locally (and the whole book serialises into the ledger JSON, so a
scheduled job can load it, take one step, and commit it back); with a
``SchwabBroker`` the very same logic places orders against a Schwab account.

This is deliberately broker-pluggable: the strategy and cadence code never know
which broker is underneath. Each :func:`step` is idempotent per trading day —
if the latest available close is one already recorded, it's a no-op.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import execution
from .broker.base import Broker
from .broker.paper import PaperBroker
from .data import DataSource
from .strategies import crisp


# How much history to pull so the strategy always has a full lookback window.
def _fetch_start(lookback: int, pad_days: int = 120) -> int:
    """Calendar days of history to request for ``lookback`` trading days."""
    # ~252 trading days per 365 calendar; pad generously for holidays/weekends.
    return int(lookback * 365 / 252) + pad_days


@dataclass
class Ledger:
    """The paper-portfolio state, serialised to JSON verbatim.

    Cash and positions live inside ``broker_state`` (a serialised
    ``PaperBroker``) for the local-simulation case; for a remote broker the book
    is the broker's own and ``broker_state`` stays ``None``.
    """

    strategy: str
    tickers: list[str]
    inception: str | None = None
    initial_capital: float = 100_000.0
    lookback: int = 252
    rebalance_every: int = 21
    slippage_bps: float = 1.0          # adverse per-fill slippage (paper broker)
    min_notional: float = 0.0          # skip rebalancing trades smaller than this

    # weights effective right now (target from the last rebalance)
    weights: dict[str, float] = field(default_factory=dict)

    last_date: str | None = None          # last close marked to market
    last_rebalance: str | None = None      # last date we traded
    days_since_rebalance: int = 0          # trading days since last trade
    rebalance_count: int = 0
    last_orders: list[dict] = field(default_factory=list)  # trades at last rebalance

    # serialised PaperBroker (cash + positions + fills); None for remote brokers
    broker_state: dict | None = None

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

    # -- paper-broker glue -------------------------------------------------- #
    def make_paper_broker(self) -> PaperBroker:
        """Reconstruct the local paper book from ``broker_state`` (or a fresh one)."""
        if self.broker_state:
            return PaperBroker.from_dict(self.broker_state)
        return PaperBroker(slippage_bps=self.slippage_bps)


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


def step(
    ledger: Ledger,
    broker: Broker,
    source: DataSource,
    *,
    as_of: str | None = None,
    allow_short: bool = True,
) -> bool:
    """Advance the ledger by at most one trading day. Returns True if it acted.

    Pulls recent prices and, for the latest close not yet recorded: rebalances
    via the broker if the cadence is due (or it's inception), marks the book to
    market, and appends a point to the equity history. Idempotent if there's no
    new close. If ``broker`` is a ``PaperBroker``, its post-step state is written
    back into ``ledger.broker_state``.
    """
    anchor = pd.Timestamp(as_of) if as_of else pd.Timestamp.today()
    start = (anchor - pd.Timedelta(days=_fetch_start(ledger.lookback))).strftime("%Y-%m-%d")
    data = source.get_prices(ledger.tickers, start=start, end=as_of)
    prices = data.prices.dropna(how="any")
    cols = [t for t in ledger.tickers if t in prices.columns]
    prices = prices[cols]
    rets = data.returns("simple")[cols].dropna(how="any")

    if len(prices) == 0:
        return False
    latest = prices.index[-1]
    latest_str = str(latest.date())

    if ledger.last_date == latest_str:        # already recorded this close
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
        ledger.tickers = cols

    if isinstance(broker, PaperBroker):
        if first_run:
            broker.ensure_funded(ledger.initial_capital)
        broker.stage_prices(last_prices)

    if not first_run:
        ledger.days_since_rebalance += 1

    due = first_run or ledger.days_since_rebalance >= ledger.rebalance_every
    if due:
        weights = {t: float(w) for t, w in zip(cols, _target_weights(ledger.strategy, window))}
        orders = execution.rebalance(
            broker, weights, last_prices,
            allow_short=allow_short, min_notional=ledger.min_notional,
        )
        ledger.weights = weights
        ledger.last_orders = [
            {"symbol": o.symbol, "side": o.side.value, "qty": o.qty} for o in orders
        ]
        ledger.last_rebalance = latest_str
        ledger.days_since_rebalance = 0
        ledger.rebalance_count += 1

    # Prefer the broker's authoritative equity (real brokers compute it
    # consistently); fall back to recomputing from positions for the paper sim.
    nav = broker.equity(last_prices)
    if nav is None:
        nav = broker.get_account().market_value(last_prices)
    ledger.last_date = latest_str
    ledger.history_dates.append(latest_str)
    ledger.history_value.append(round(nav, 2))
    ledger.history_equity.append(round(nav / ledger.initial_capital, 6))

    if isinstance(broker, PaperBroker):
        ledger.broker_state = broker.to_dict()
    return True
