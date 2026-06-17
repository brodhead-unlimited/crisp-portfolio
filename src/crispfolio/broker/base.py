"""Broker abstraction — interface plus the value types orders flow through.

Mirrors the ``data/`` package idiom: an ABC (:class:`Broker`) plus small,
JSON-friendly dataclasses (:class:`Order`, :class:`Fill`, :class:`Position`,
:class:`Account`), with concrete implementations (``PaperBroker``,
``SchwabBroker``) living alongside and re-exported from ``broker/__init__``.

The point of the seam: the portfolio engine speaks only in target weights and
reads/writes positions through this interface, so the *same* rebalance logic
drives a self-simulated paper book (``PaperBroker``) or a real Schwab account
(``SchwabBroker``) with no change to the strategy code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Side(str, Enum):
    """Equity order instructions, matching Schwab's vocabulary.

    Schwab requires the *direction relative to the existing position* to be
    explicit: opening/adding a long is ``BUY``, reducing/closing it is
    ``SELL``; opening/adding a short is ``SELL_SHORT``, reducing/closing it is
    ``BUY_TO_COVER``. The execution layer is responsible for choosing the right
    one given the current position.
    """

    BUY = "BUY"
    SELL = "SELL"
    SELL_SHORT = "SELL_SHORT"
    BUY_TO_COVER = "BUY_TO_COVER"

    @property
    def signed(self) -> int:
        """+1 if the instruction adds shares to the book, -1 if it removes."""
        return 1 if self in (Side.BUY, Side.BUY_TO_COVER) else -1


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class Order:
    """An instruction to trade ``qty`` (always positive) shares of ``symbol``."""

    symbol: str
    side: Side
    qty: int
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    tif: str = "DAY"
    client_order_id: str | None = None

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"order qty must be positive, got {self.qty}")
        # allow plain strings from JSON/callers
        self.side = Side(self.side)
        self.type = OrderType(self.type)


@dataclass
class Fill:
    """A (partial or full) execution of an order at ``price``."""

    symbol: str
    side: Side
    qty: int
    price: float
    commission: float = 0.0
    ts: str | None = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "qty": self.qty,
            "price": round(self.price, 6),
            "commission": round(self.commission, 6),
            "ts": self.ts,
        }


@dataclass
class Position:
    """A signed holding: ``qty`` < 0 is a short."""

    symbol: str
    qty: int
    avg_price: float = 0.0

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "qty": self.qty,
                "avg_price": round(self.avg_price, 6)}

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(symbol=d["symbol"], qty=int(d["qty"]),
                   avg_price=float(d.get("avg_price", 0.0)))


@dataclass
class Account:
    """A point-in-time snapshot of investable cash plus open positions."""

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    def market_value(self, prices: pd.Series) -> float:
        """Net liquidation value = cash + Σ qty·price over priced positions.

        Shorts contribute negative position value, which nets against the cash
        their proceeds added — so a freshly-opened short leaves NAV unchanged
        (modulo costs), exactly as a real margin book does.
        """
        held = 0.0
        for sym, pos in self.positions.items():
            if sym in prices.index:
                held += pos.qty * float(prices[sym])
        return self.cash + held


class Broker(ABC):
    """Anything that can hold positions and execute orders."""

    @abstractmethod
    def get_account(self) -> Account:
        """Return current cash + positions."""
        raise NotImplementedError

    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Submit ``order``; return a broker order id."""
        raise NotImplementedError

    @abstractmethod
    def get_order(self, order_id: str) -> dict:
        """Return the broker's status payload for ``order_id``."""
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel a working order."""
        raise NotImplementedError

    def stage_prices(self, prices: pd.Series) -> None:
        """Provide reference marks for the next fills.

        Brokers with their own market data (e.g. Schwab) ignore this; the paper
        broker uses it to price synchronous fills. Default: no-op.
        """
        return None
