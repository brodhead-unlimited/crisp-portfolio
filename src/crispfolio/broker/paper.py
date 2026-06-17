"""``PaperBroker`` — a self-contained simulated broker.

Fills ``MARKET`` orders synchronously at staged reference prices, nudged
adversely by ``slippage_bps`` (buys fill a hair high, sells a hair low), plus an
optional per-fill commission. It keeps whole-share, sign-aware positions and a
cash balance using ordinary double-entry, so opening a short adds the sale
proceeds to cash while the negative position value nets it back out.

State is plain JSON (``to_dict``/``from_dict``) so the daily job can load the
book from the ledger, take one step, and commit it back. This broker has no
network, no auth, and no real money, which is exactly why it can drive the
public website unattended.

Order sizing and the weights→orders translation live in
:mod:`crispfolio.execution`; this class only knows how to fill a single order
and report the book.
"""
from __future__ import annotations

import pandas as pd

from .base import Account, Broker, Fill, Order, Position, Side


class PaperBroker(Broker):
    def __init__(
        self,
        cash: float = 0.0,
        positions: dict[str, Position] | None = None,
        fills: list[Fill] | None = None,
        slippage_bps: float = 1.0,
        commission: float = 0.0,
        next_id: int = 1,
    ):
        self.cash = float(cash)
        self._positions: dict[str, Position] = positions or {}
        self.fills: list[Fill] = fills or []
        self.slippage_bps = float(slippage_bps)
        self.commission = float(commission)
        self._ref: pd.Series | None = None
        self._next_id = next_id

    # -- funding / pricing -------------------------------------------------- #
    def ensure_funded(self, capital: float) -> None:
        """Seed the book with ``capital`` cash if it has never traded."""
        if not self._positions and self.cash == 0.0:
            self.cash = float(capital)

    def stage_prices(self, prices: pd.Series) -> None:
        self._ref = prices

    # -- Broker interface --------------------------------------------------- #
    def get_account(self) -> Account:
        return Account(cash=self.cash, positions=dict(self._positions))

    def place_order(self, order: Order) -> str:
        if self._ref is None:
            raise RuntimeError("no reference prices staged; call stage_prices() first")
        if order.symbol not in self._ref.index:
            raise RuntimeError(f"no reference price for {order.symbol!r}")
        ref = float(self._ref[order.symbol])
        slip = self.slippage_bps / 1e4
        adds = order.side.signed > 0  # BUY / BUY_TO_COVER lift the fill price
        fill_px = ref * (1 + slip) if adds else ref * (1 - slip)

        signed_qty = order.side.signed * order.qty
        # buying spends cash (-), selling/shorting raises cash (+); commission
        # always costs.
        self.cash += -signed_qty * fill_px - self.commission
        self._apply_position(order.symbol, signed_qty, fill_px)

        self.fills.append(
            Fill(order.symbol, order.side, order.qty, fill_px, self.commission)
        )
        oid = str(self._next_id)
        self._next_id += 1
        return oid

    def get_order(self, order_id: str) -> dict:
        # Paper orders fill instantly; expose the matching fill if present.
        return {"order_id": order_id, "status": "FILLED"}

    def cancel_order(self, order_id: str) -> None:
        return None

    # -- internals ---------------------------------------------------------- #
    def _apply_position(self, symbol: str, signed_qty: int, px: float) -> None:
        pos = self._positions.get(symbol)
        old = pos.qty if pos else 0
        new = old + signed_qty
        if new == 0:
            self._positions.pop(symbol, None)
            return
        if pos is None or old == 0:
            avg = px
        elif (old > 0) == (signed_qty > 0):
            # adding in the same direction -> magnitude-weighted average cost
            avg = (abs(old) * pos.avg_price + abs(signed_qty) * px) / abs(new)
        else:
            # reducing toward zero (never crossing — execution splits flips)
            avg = pos.avg_price
        self._positions[symbol] = Position(symbol, new, avg)

    # -- (de)serialisation -------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "cash": round(self.cash, 6),
            "positions": [p.to_dict() for p in self._positions.values()],
            "fills": [f.to_dict() for f in self.fills],
            "slippage_bps": self.slippage_bps,
            "commission": self.commission,
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PaperBroker":
        positions = {
            p["symbol"]: Position.from_dict(p) for p in d.get("positions", [])
        }
        fills = [
            Fill(
                symbol=f["symbol"],
                side=Side(f["side"]),
                qty=int(f["qty"]),
                price=float(f["price"]),
                commission=float(f.get("commission", 0.0)),
                ts=f.get("ts"),
            )
            for f in d.get("fills", [])
        ]
        return cls(
            cash=float(d.get("cash", 0.0)),
            positions=positions,
            fills=fills,
            slippage_bps=float(d.get("slippage_bps", 1.0)),
            commission=float(d.get("commission", 0.0)),
            next_id=int(d.get("next_id", 1)),
        )
