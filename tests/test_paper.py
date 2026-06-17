"""Tests for the paper-trading engine, driven by a synthetic price source.

The engine now routes orders through a ``PaperBroker``; each step rebuilds the
broker from the ledger (exactly as the scheduled job does), so these tests also
exercise broker-state persistence end to end.
"""
import numpy as np
import pandas as pd
import pytest

from crispfolio.data.base import DataSource, PriceData
from crispfolio.paper import Ledger, step


class SyntheticSource(DataSource):
    """Deterministic geometric-random-walk prices, sliced by [start, end]."""

    def __init__(self, tickers, n_days=400, seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2022-01-03", periods=n_days)
        rets = rng.normal(0.0003, 0.01, size=(n_days, len(tickers)))
        prices = 100 * np.exp(np.cumsum(rets, axis=0))
        self._panel = pd.DataFrame(prices, index=idx, columns=tickers)

    def get_prices(self, tickers, start, end=None) -> PriceData:
        df = self._panel.loc[:, [t for t in tickers if t in self._panel.columns]]
        df = df.loc[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df.loc[df.index <= pd.Timestamp(end)]
        return PriceData(df.copy())


@pytest.fixture
def tickers():
    return [f"A{i}" for i in range(6)]


@pytest.fixture
def src(tickers):
    return SyntheticSource(tickers, n_days=400, seed=1)


def _new_ledger(tickers):
    return Ledger(strategy="crisp", tickers=tickers, lookback=120,
                  rebalance_every=21, initial_capital=100_000.0)


def _run(led, src, as_of):
    """Rebuild the broker from the ledger (as the job does), then step once."""
    broker = led.make_paper_broker()
    return step(led, broker, src, as_of=as_of)


def test_first_step_initialises_and_invests(src, tickers):
    led = _new_ledger(tickers)
    acted = _run(led, src, as_of="2023-01-15")
    assert acted is True
    assert led.inception is not None
    assert led.rebalance_count == 1            # inception trades
    assert led.last_date is not None
    assert led.broker_state is not None        # paper book persisted
    assert led.last_orders                      # placed at least one order
    # CRISP is long-short (gross normalisation): |weights| sum to ~1.
    assert abs(sum(abs(w) for w in led.weights.values()) - 1.0) < 1e-6
    # whole-share rounding + slippage keeps NAV just below the starting capital
    eq = led.history_equity[-1]
    assert 0.95 < eq <= 1.0
    assert eq == pytest.approx(led.history_value[-1] / led.initial_capital, rel=1e-6)


def test_book_holds_short_legs(src, tickers):
    led = _new_ledger(tickers)
    _run(led, src, as_of="2023-01-15")
    qtys = [p["qty"] for p in led.broker_state["positions"]]
    assert any(q < 0 for q in qtys)            # CRISP shorts are really shorted
    assert all(float(q).is_integer() for q in qtys)  # whole shares only


def test_idempotent_same_day(src, tickers):
    led = _new_ledger(tickers)
    _run(led, src, as_of="2023-01-15")
    n = len(led.history_dates)
    acted = _run(led, src, as_of="2023-01-15")   # same close again
    assert acted is False
    assert len(led.history_dates) == n           # no duplicate point


def test_rebalances_only_on_cadence(src, tickers):
    led = _new_ledger(tickers)
    for d in pd.bdate_range("2023-01-16", periods=50):
        _run(led, src, as_of=d.strftime("%Y-%m-%d"))
    assert led.rebalance_count >= 2
    assert led.rebalance_count < 10
    assert len(led.history_dates) >= 40


def test_roundtrip_persistence(tmp_path, src, tickers):
    led = _new_ledger(tickers)
    _run(led, src, as_of="2023-02-01")
    p = tmp_path / "ledger.json"
    led.save(p)
    again = Ledger.load(p)
    assert again is not None
    assert again.last_date == led.last_date
    assert again.weights == led.weights
    assert again.history_equity == led.history_equity
    assert again.broker_state == led.broker_state
    # resuming from disk does not double-count the same day
    assert _run(again, src, as_of=led.last_date) is False


def test_equity_tracks_value(src, tickers):
    led = _new_ledger(tickers)
    for d in pd.bdate_range("2023-01-16", periods=30):
        _run(led, src, as_of=d.strftime("%Y-%m-%d"))
    for v, e in zip(led.history_value, led.history_equity):
        assert e == pytest.approx(v / led.initial_capital, rel=1e-4)
