"""Tests for the PaperBroker fill simulation."""
import pandas as pd
import pytest

from crispfolio.broker.base import Order, Side
from crispfolio.broker.paper import PaperBroker


@pytest.fixture
def prices():
    return pd.Series({"AAA": 100.0, "BBB": 50.0})


def _broker(prices, **kw):
    b = PaperBroker(**kw)
    b.ensure_funded(100_000.0)
    b.stage_prices(prices)
    return b


def test_buy_reduces_cash_and_adds_position(prices):
    b = _broker(prices, slippage_bps=0.0)
    b.place_order(Order("AAA", Side.BUY, 100))
    acct = b.get_account()
    assert acct.positions["AAA"].qty == 100
    assert acct.cash == pytest.approx(100_000 - 100 * 100.0)
    # marked at the same price, NAV is unchanged with zero slippage
    assert acct.market_value(prices) == pytest.approx(100_000.0)


def test_slippage_costs_on_buy(prices):
    b = _broker(prices, slippage_bps=10.0)  # 10 bps
    b.place_order(Order("AAA", Side.BUY, 100))
    acct = b.get_account()
    # fill 0.1% above 100 -> NAV drops by the slippage paid
    assert acct.market_value(prices) == pytest.approx(100_000 - 100 * 100 * 1e-3)


def test_short_then_cover_conserves_cash(prices):
    b = _broker(prices, slippage_bps=0.0)
    b.place_order(Order("AAA", Side.SELL_SHORT, 50))
    acct = b.get_account()
    assert acct.positions["AAA"].qty == -50
    assert acct.cash == pytest.approx(100_000 + 50 * 100.0)   # proceeds in
    assert acct.market_value(prices) == pytest.approx(100_000.0)  # NAV unchanged
    # cover the whole short -> flat, cash back to start
    b.place_order(Order("AAA", Side.BUY_TO_COVER, 50))
    acct = b.get_account()
    assert "AAA" not in acct.positions
    assert acct.cash == pytest.approx(100_000.0)


def test_average_price_on_add(prices):
    b = _broker(prices, slippage_bps=0.0)
    b.place_order(Order("AAA", Side.BUY, 100))
    b.stage_prices(pd.Series({"AAA": 120.0, "BBB": 50.0}))
    b.place_order(Order("AAA", Side.BUY, 100))
    pos = b.get_account().positions["AAA"]
    assert pos.qty == 200
    assert pos.avg_price == pytest.approx(110.0)   # (100*100 + 100*120)/200


def test_roundtrip_serialisation(prices):
    b = _broker(prices, slippage_bps=2.0)
    b.place_order(Order("AAA", Side.BUY, 30))
    b.place_order(Order("BBB", Side.SELL_SHORT, 10))
    restored = PaperBroker.from_dict(b.to_dict())
    a1, a2 = b.get_account(), restored.get_account()
    assert a1.cash == pytest.approx(a2.cash)
    assert {s: p.qty for s, p in a1.positions.items()} == \
           {s: p.qty for s, p in a2.positions.items()}
    assert len(restored.fills) == 2


def test_place_without_price_raises():
    b = PaperBroker()
    b.ensure_funded(100_000.0)
    with pytest.raises(RuntimeError):
        b.place_order(Order("AAA", Side.BUY, 1))   # no staged prices
