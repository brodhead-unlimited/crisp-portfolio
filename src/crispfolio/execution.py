"""Translate target weights into broker orders — the weights→trades seam.

The strategy emits a target weight per ticker; a brokerage account holds an
integer number of shares. This module bridges the two:

* :func:`target_positions` sizes whole-share targets off live NAV.
* :func:`diff_to_orders` turns a (current → target) position change into the
  minimal set of correctly-instructed orders, splitting any long↔short flip
  into a close + an open so no single order crosses zero.
* :func:`rebalance` runs the whole cycle against a :class:`Broker`.

Keeping this broker-agnostic means the paper book and a real Schwab account
rebalance through identical code.
"""
from __future__ import annotations

import pandas as pd

from .broker.base import Broker, Fill, Order, Side


def target_positions(
    weights: dict[str, float],
    nav: float,
    prices: pd.Series,
    *,
    allow_short: bool = True,
) -> dict[str, int]:
    """Whole-share target per ticker for the given NAV and prices.

    Dollars are truncated toward zero into shares, so the book never
    over-commits cash to rounding. Sign is preserved (negative = short). With
    ``allow_short=False`` negative weights are dropped to zero.
    """
    out: dict[str, int] = {}
    for sym, w in weights.items():
        if not allow_short and w < 0:
            continue
        if sym not in prices.index:
            continue
        px = float(prices[sym])
        if px <= 0:
            continue
        qty = int(nav * w / px)  # int() truncates toward zero, keeping sign
        if qty != 0:
            out[sym] = qty
    return out


def diff_to_orders(
    current: dict[str, int],
    target: dict[str, int],
    prices: pd.Series,
    *,
    min_notional: float = 0.0,
) -> list[Order]:
    """Minimal orders to move ``current`` holdings to ``target``.

    A long→short (or short→long) move is emitted as two orders — first closing
    the existing side, then opening the new one — because a broker order cannot
    cross through zero in one instruction. Orders whose notional is below
    ``min_notional`` are skipped to avoid churning on dust.
    """
    orders: list[Order] = []
    for sym in sorted(set(current) | set(target)):
        cur = int(current.get(sym, 0))
        tgt = int(target.get(sym, 0))
        if cur == tgt:
            continue
        px = float(prices[sym]) if sym in prices.index else 0.0

        for side, qty in _legs(cur, tgt):
            if px and qty * px < min_notional:
                continue
            orders.append(Order(symbol=sym, side=side, qty=qty))
    return orders


def _legs(cur: int, tgt: int) -> list[tuple[Side, int]]:
    """The (side, qty) legs to go from ``cur`` to ``tgt`` shares."""
    # Crossing zero: close the old side fully, then open the new side.
    if cur > 0 and tgt < 0:
        return [(Side.SELL, cur), (Side.SELL_SHORT, -tgt)]
    if cur < 0 and tgt > 0:
        return [(Side.BUY_TO_COVER, -cur), (Side.BUY, tgt)]

    # Same side (or from/to flat).
    delta = tgt - cur
    if cur >= 0 and tgt >= 0:
        return [(Side.BUY, delta)] if delta > 0 else [(Side.SELL, -delta)]
    # both <= 0
    return [(Side.SELL_SHORT, -delta)] if delta < 0 else [(Side.BUY_TO_COVER, delta)]


def rebalance(
    broker: Broker,
    weights: dict[str, float],
    prices: pd.Series,
    *,
    allow_short: bool = True,
    min_notional: float = 0.0,
) -> list[Order]:
    """Size targets off live NAV, diff against the book, and place the orders.

    Returns the orders submitted (for reporting). Resulting positions/cash are
    read back from ``broker.get_account()`` by the caller.
    """
    acct = broker.get_account()
    nav = acct.market_value(prices)
    current = {sym: pos.qty for sym, pos in acct.positions.items()}
    target = target_positions(weights, nav, prices, allow_short=allow_short)
    orders = diff_to_orders(current, target, prices, min_notional=min_notional)
    for order in orders:
        broker.place_order(order)
    return orders
