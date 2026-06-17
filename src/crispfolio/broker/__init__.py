"""Broker layer: order execution behind a single interface.

``PaperBroker`` simulates fills locally (drives the public paper portfolio);
``SchwabBroker`` places real orders through Schwab's Trader API (sandbox or,
behind hard guards, production). Both satisfy :class:`Broker`, so the rebalance
logic in :mod:`crispfolio.execution` is identical for either.
"""
from .base import Account, Broker, Fill, Order, OrderType, Position, Side
from .alpaca import AlpacaBroker
from .paper import PaperBroker
from .schwab import SchwabBroker

__all__ = [
    "Account",
    "Broker",
    "Fill",
    "Order",
    "OrderType",
    "Position",
    "Side",
    "AlpacaBroker",
    "PaperBroker",
    "SchwabBroker",
]
