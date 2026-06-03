"""Free public price data via yfinance, with a tiny on-disk cache.

Used for research and backtesting.  For live prices / real holdings on the
website, write a SchwabDataSource implementing the same ``DataSource``
interface and the rest of the code is unchanged.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .base import DataSource, PriceData

_CACHE_DIR = Path("data_cache")


class YFinanceSource(DataSource):
    def __init__(self, cache: bool = True, cache_dir: Path | str = _CACHE_DIR):
        self.cache = cache
        self.cache_dir = Path(cache_dir)
        if self.cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, tickers, start, end) -> Path:
        key = "|".join(sorted(tickers)) + f"|{start}|{end}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"yf_{digest}.pkl"

    def get_prices(self, tickers, start, end=None) -> PriceData:
        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        path = self._cache_path(tickers, start, end)
        if self.cache and path.exists():
            return PriceData(pd.read_pickle(path))

        import yfinance as yf

        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        # yfinance returns a column MultiIndex (field, ticker) for >1 ticker.
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy()
        else:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

        prices = prices.dropna(how="all").sort_index()
        # keep only tickers that actually returned data
        prices = prices.dropna(axis=1, how="all")
        if self.cache:
            prices.to_pickle(path)
        return PriceData(prices)
