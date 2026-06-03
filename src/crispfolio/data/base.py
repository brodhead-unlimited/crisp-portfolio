"""Data-source abstraction.

The whole library only ever needs a panel of adjusted close prices
(dates x tickers).  Keeping that behind a small interface means the
backtest/research code is identical whether prices come from a free
public feed (yfinance) or a brokerage API (Schwab) later on.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PriceData:
    """Adjusted close prices and the returns derived from them."""

    prices: pd.DataFrame  # index: DatetimeIndex, columns: tickers

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    def returns(self, kind: str = "log") -> pd.DataFrame:
        """Periodic returns. ``kind`` is 'log' or 'simple'."""
        if kind == "log":
            r = np.log(self.prices).diff()
        elif kind == "simple":
            r = self.prices.pct_change()
        else:
            raise ValueError(f"unknown return kind: {kind!r}")
        return r.dropna(how="all")


class DataSource(ABC):
    """Anything that can hand back a price panel for a set of tickers."""

    @abstractmethod
    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
    ) -> PriceData:
        """Return adjusted-close prices for ``tickers`` over [start, end]."""
        raise NotImplementedError
