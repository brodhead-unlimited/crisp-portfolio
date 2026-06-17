"""Tests for the weights→orders translation (crispfolio.execution)."""
import pandas as pd
import pytest

from crispfolio.broker.base import Side
from crispfolio.execution import diff_to_orders, target_positions


@pytest.fixture
def prices():
    return pd.Series({"AAA": 100.0, "BBB": 50.0, "CCC": 25.0})


def test_target_positions_whole_shares_and_sign(prices):
    weights = {"AAA": 0.5, "BBB": -0.25, "CCC": 0.0}
    tgt = target_positions(weights, nav=100_000.0, prices=prices)
    assert tgt["AAA"] == 500          # 50_000 / 100
    assert tgt["BBB"] == -500         # -25_000 / 50, sign preserved
    assert "CCC" not in tgt           # zero target dropped


def test_target_positions_truncates_toward_zero(prices):
    # 0.3333 * 30_000 = ... deliberately non-integer share count
    tgt = target_positions({"CCC": 0.3334}, nav=10_000.0, prices=prices)
    assert tgt["CCC"] == int(10_000 * 0.3334 / 25.0)  # truncated, not rounded


def test_target_positions_no_short(prices):
    tgt = target_positions({"AAA": 0.5, "BBB": -0.5}, nav=100_000.0,
                           prices=prices, allow_short=False)
    assert tgt == {"AAA": 500}


def test_diff_simple_buy_and_sell(prices):
    orders = diff_to_orders({"AAA": 100}, {"AAA": 250, "BBB": -40}, prices)
    by_sym = {(o.symbol, o.side): o.qty for o in orders}
    assert by_sym[("AAA", Side.BUY)] == 150
    assert by_sym[("BBB", Side.SELL_SHORT)] == 40


def test_diff_long_to_short_splits(prices):
    # +5 -> -3 must close the long then open the short (no zero-crossing order)
    orders = diff_to_orders({"AAA": 5}, {"AAA": -3}, prices)
    assert [(o.side, o.qty) for o in orders] == [
        (Side.SELL, 5), (Side.SELL_SHORT, 3),
    ]


def test_diff_short_to_long_splits(prices):
    orders = diff_to_orders({"AAA": -4}, {"AAA": 6}, prices)
    assert [(o.side, o.qty) for o in orders] == [
        (Side.BUY_TO_COVER, 4), (Side.BUY, 6),
    ]


def test_diff_reduce_short_covers(prices):
    # -10 -> -4 reduces the short: buy to cover 6
    orders = diff_to_orders({"AAA": -10}, {"AAA": -4}, prices)
    assert [(o.side, o.qty) for o in orders] == [(Side.BUY_TO_COVER, 6)]


def test_diff_min_notional_skips_dust(prices):
    # CCC delta is 1 share * $25 = $25 notional; threshold $100 skips it
    orders = diff_to_orders({}, {"AAA": 10, "CCC": 1}, prices, min_notional=100.0)
    syms = {o.symbol for o in orders}
    assert syms == {"AAA"}


def test_diff_noop_when_equal(prices):
    assert diff_to_orders({"AAA": 100}, {"AAA": 100}, prices) == []
