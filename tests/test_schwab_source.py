"""Offline tests for SchwabDataSource — the HTTP layer is faked.

Covers epoch conversion, single/multi symbol panel assembly, empty-candle
handling, and the all-empty error path. A cached, non-expired token is written
so ``get_access_token`` never hits the network.
"""
import json

import pandas as pd
import pytest

from crispfolio.config import SchwabConfig
from crispfolio import schwab_auth as auth
from crispfolio.data.schwab_source import SchwabDataSource, _to_epoch_ms


@pytest.fixture
def cfg(tmp_path):
    return SchwabConfig(
        app_key="APPKEY",
        app_secret="APPSECRET",
        callback_url="https://127.0.0.1:8182",
        token_path=tmp_path / "schwab_token.json",
    )


@pytest.fixture
def authed_cfg(cfg):
    # expiry must land beyond real wall-clock time (get_access_token uses
    # time.time()), so no refresh/network is attempted.
    tok = auth.Token(
        access_token="acc",
        refresh_token="ref",
        expires_in=10**12,
        obtained_at=0.0,
    )
    auth.save_token(cfg, tok)
    return cfg


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _FakeSession:
    """Serves candles keyed by the requested ``symbol`` param."""

    def __init__(self, candles_by_symbol, status_code=200):
        self._candles = candles_by_symbol
        self._status = status_code
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(params)
        sym = params["symbol"]
        return _FakeResponse(
            {"symbol": sym, "candles": self._candles.get(sym, []), "empty": False},
            status_code=self._status,
        )


def _candle(date_str, close):
    ms = int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)
    return {"datetime": ms, "open": close, "high": close, "low": close,
            "close": close, "volume": 100}


def test_to_epoch_ms_roundtrip():
    ms = _to_epoch_ms("2020-01-02")
    assert pd.Timestamp(ms, unit="ms", tz="UTC") == pd.Timestamp("2020-01-02", tz="UTC")


def test_single_symbol_panel(authed_cfg):
    session = _FakeSession({
        "XLK": [_candle("2020-01-02", 100.0), _candle("2020-01-03", 101.0)],
    })
    src = SchwabDataSource(cfg=authed_cfg, cache=False, session=session)
    pd_data = src.get_prices(["XLK"], start="2020-01-01", end="2020-02-01")
    assert list(pd_data.prices.columns) == ["XLK"]
    assert len(pd_data.prices) == 2
    assert pd_data.prices["XLK"].iloc[-1] == 101.0
    # periodType must be daily-capable (a bare periodType=day rejects daily)
    assert session.calls[0]["periodType"] == "year"
    assert session.calls[0]["frequencyType"] == "daily"


def test_multi_symbol_alignment(authed_cfg):
    session = _FakeSession({
        "XLK": [_candle("2020-01-02", 100.0), _candle("2020-01-03", 101.0)],
        "TLT": [_candle("2020-01-03", 50.0)],  # missing 01-02 -> NaN after align
    })
    src = SchwabDataSource(cfg=authed_cfg, cache=False, session=session)
    prices = src.get_prices(["XLK", "TLT"], start="2020-01-01").prices
    assert set(prices.columns) == {"XLK", "TLT"}
    assert prices.loc["2020-01-02", "TLT"] != prices.loc["2020-01-02", "TLT"]  # NaN
    assert prices.loc["2020-01-03", "TLT"] == 50.0


def test_empty_symbol_dropped(authed_cfg):
    session = _FakeSession({
        "XLK": [_candle("2020-01-02", 100.0)],
        "BAD": [],
    })
    src = SchwabDataSource(cfg=authed_cfg, cache=False, session=session)
    prices = src.get_prices(["XLK", "BAD"], start="2020-01-01").prices
    assert list(prices.columns) == ["XLK"]


def test_all_empty_raises(authed_cfg):
    session = _FakeSession({})
    src = SchwabDataSource(cfg=authed_cfg, cache=False, session=session)
    with pytest.raises(RuntimeError):
        src.get_prices(["NOPE"], start="2020-01-01")


def test_http_error_raises(authed_cfg):
    session = _FakeSession({"XLK": []}, status_code=401)
    src = SchwabDataSource(cfg=authed_cfg, cache=False, session=session)
    with pytest.raises(RuntimeError):
        src.get_prices(["XLK"], start="2020-01-01")
