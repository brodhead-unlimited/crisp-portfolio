"""Tests for the paper-trading engine, driven by a synthetic price source."""
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


def test_first_step_initialises_and_invests(src, tickers):
    led = _new_ledger(tickers)
    acted = step(led, src, as_of="2023-01-15")
    assert acted is True
    assert led.inception is not None
    assert led.rebalance_count == 1            # inception trades
    assert led.last_date is not None
    # CRISP is long-short (gross normalisation): |weights| sum to ~1.
    # (Net signed cash can be large — shorts add cash — so it is not a
    # rounding residual here; the meaningful invariant is NAV below.)
    assert abs(sum(abs(w) for w in led.weights.values()) - 1.0) < 1e-6
    # inception charges one turnover (all-cash -> invested) at cost_bps,
    # so first equity ~= 1 - cost_bps/1e4, not exactly 1.0
    assert led.history_equity[-1] == pytest.approx(1.0 - led.cost_bps / 1e4, abs=1e-4)


def test_idempotent_same_day(src, tickers):
    led = _new_ledger(tickers)
    step(led, src, as_of="2023-01-15")
    n = len(led.history_dates)
    acted = step(led, src, as_of="2023-01-15")   # same close again
    assert acted is False
    assert len(led.history_dates) == n           # no duplicate point


def test_rebalances_only_on_cadence(src, tickers):
    led = _new_ledger(tickers)
    # walk day by day across ~50 business days
    days = pd.bdate_range("2023-01-16", periods=50)
    for d in days:
        step(led, src, as_of=d.strftime("%Y-%m-%d"))
    # inception + at least one cadence rebalance, but far fewer than #days
    assert led.rebalance_count >= 2
    assert led.rebalance_count < 10
    # history grew roughly one point per business day stepped
    assert len(led.history_dates) >= 40


def test_roundtrip_persistence(tmp_path, src, tickers):
    led = _new_ledger(tickers)
    step(led, src, as_of="2023-02-01")
    p = tmp_path / "ledger.json"
    led.save(p)
    again = Ledger.load(p)
    assert again is not None
    assert again.last_date == led.last_date
    assert again.weights == led.weights
    assert again.history_equity == led.history_equity
    # resuming from disk does not double-count the same day
    assert step(again, src, as_of=led.last_date) is False


def test_equity_tracks_value(src, tickers):
    led = _new_ledger(tickers)
    for d in pd.bdate_range("2023-01-16", periods=30):
        step(led, src, as_of=d.strftime("%Y-%m-%d"))
    # equity is value / initial_capital, elementwise
    for v, e in zip(led.history_value, led.history_equity):
        assert e == pytest.approx(v / led.initial_capital, rel=1e-4)
