"""Schwab market-data price source — a drop-in ``DataSource``.

Implements the same ``get_prices`` contract as ``YFinanceSource`` so the
backtest/research code is unchanged: swap ``YFinanceSource()`` for
``SchwabDataSource()`` and everything downstream behaves identically.

Authentication is delegated to ``crispfolio.schwab_auth`` (OAuth token in the
macOS Keychain + cached refresh token). Run ``scripts/schwab_login.py`` once
before using this source.

Endpoint (Schwab market-data API):
    GET https://api.schwabapi.com/marketdata/v1/pricehistory
    params: symbol, frequencyType=daily, frequency=1,
            startDate / endDate as epoch-millisecond integers
    returns: {"candles": [{"close": float, "datetime": <epoch ms>, ...}], ...}

One symbol per request, so a multi-ticker panel is assembled column by column.

Caveat on adjustment: ``yfinance`` is queried with ``auto_adjust=True`` (splits
*and* dividends folded into the close). Schwab's ``pricehistory`` close is
split-adjusted but **not** dividend-adjusted. For total-return backtests the
two sources are therefore not identical on dividend-paying assets; treat
Schwab as the live/holdings feed and keep yfinance for long historical
backtests, or reconcile the adjustment explicitly before mixing them.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd
import requests

from ..config import SchwabConfig
from ..schwab_auth import get_access_token
from .base import DataSource, PriceData

PRICE_HISTORY_URL = "https://api.schwabapi.com/marketdata/v1/pricehistory"
_CACHE_DIR = Path("data_cache")


def _to_epoch_ms(date_str: str) -> int:
    """Convert a 'YYYY-MM-DD' string to integer epoch milliseconds (UTC)."""
    ts = pd.Timestamp(date_str, tz="UTC")
    return int(ts.timestamp() * 1000)


class SchwabDataSource(DataSource):
    """Adjusted-close price panels from the Schwab market-data API.

    Parameters
    ----------
    cfg:
        Loaded ``SchwabConfig``; defaults to ``SchwabConfig.load()`` (reads the
        App Key / Secret from the Keychain).
    cache:
        If true, cache each assembled panel on disk (same scheme as
        ``YFinanceSource``). Useful for repeated backtests; turn off for live
        use where you want fresh prices.
    session:
        Optional ``requests.Session`` (injected in tests).
    """

    def __init__(
        self,
        cfg: SchwabConfig | None = None,
        cache: bool = True,
        cache_dir: Path | str = _CACHE_DIR,
        session: requests.Session | None = None,
    ):
        self.cfg = cfg or SchwabConfig.load()
        self.cache = cache
        self.cache_dir = Path(cache_dir)
        self._session = session or requests.Session()
        if self.cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, tickers, start, end) -> Path:
        key = "|".join(sorted(tickers)) + f"|{start}|{end}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"schwab_{digest}.pkl"

    def _fetch_symbol(self, symbol: str, start_ms: int, end_ms: int) -> pd.Series:
        """Return a daily close Series (DatetimeIndex) for one symbol."""
        token = get_access_token(self.cfg)
        resp = self._session.get(
            PRICE_HISTORY_URL,
            headers={"Authorization": f"Bearer {token}"},
            params={
                "symbol": symbol,
                # periodType must permit a daily frequency: the default
                # periodType=day only allows frequencyType=minute. With explicit
                # startDate/endDate the date window overrides the period length,
                # but periodType still gates which frequencyType is legal.
                "periodType": "year",
                "frequencyType": "daily",
                "frequency": 1,
                "startDate": start_ms,
                "endDate": end_ms,
                "needExtendedHoursData": "false",
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Schwab pricehistory for {symbol!r} returned "
                f"{resp.status_code}: {resp.text}"
            )
        payload = resp.json()
        candles = payload.get("candles", [])
        if not candles:
            return pd.Series(dtype="float64", name=symbol)
        idx = pd.to_datetime([c["datetime"] for c in candles], unit="ms")
        closes = [c["close"] for c in candles]
        return pd.Series(closes, index=idx, name=symbol).sort_index()

    def get_prices(self, tickers, start, end=None) -> PriceData:
        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        path = self._cache_path(tickers, start, end)
        if self.cache and path.exists():
            return PriceData(pd.read_pickle(path))

        start_ms, end_ms = _to_epoch_ms(start), _to_epoch_ms(end)
        cols = {}
        for symbol in tickers:
            series = self._fetch_symbol(symbol, start_ms, end_ms)
            if not series.empty:
                cols[symbol] = series

        if not cols:
            raise RuntimeError(
                "Schwab returned no price data for any requested ticker "
                f"({tickers}); check the symbols and that login is current."
            )

        # align on the union of trading days; normalise the index to dates
        prices = pd.DataFrame(cols).sort_index()
        prices.index = prices.index.normalize()
        prices = prices[~prices.index.duplicated(keep="last")]
        prices = prices.dropna(how="all").dropna(axis=1, how="all")
        if self.cache:
            prices.to_pickle(path)
        return PriceData(prices)
