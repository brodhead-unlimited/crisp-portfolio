"""Offline tests for SchwabBroker — the HTTP layer and auth are faked.

Mirrors tests/test_schwab_source.py: a cached non-expiring token keeps
``get_access_token`` off the network, and an injected fake session records the
requests and serves canned responses.
"""
import json

import pytest

from crispfolio import schwab_auth as auth
from crispfolio.broker.base import Order, OrderType, Side
from crispfolio.broker.schwab import SchwabBroker
from crispfolio.config import SchwabConfig


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
    tok = auth.Token(access_token="acc", refresh_token="ref",
                     expires_in=10**12, obtained_at=0.0)
    auth.save_token(cfg, tok)
    return cfg


class _FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload) if payload is not None else ""
        self.headers = headers or {}

    def json(self):
        return self._payload


class _FakeSession:
    """Routes by URL suffix; records every call on ``self.calls``."""

    ACCOUNTS = [{"accountNumber": "12345678", "hashValue": "HASH123"}]
    ACCOUNT_DETAIL = {
        "securitiesAccount": {
            "currentBalances": {"cashBalance": 50_000.0},
            "positions": [
                {"instrument": {"symbol": "XLK"}, "longQuantity": 100.0,
                 "shortQuantity": 0.0, "averagePrice": 200.0},
                {"instrument": {"symbol": "XLF"}, "longQuantity": 0.0,
                 "shortQuantity": 40.0, "averagePrice": 35.0},
            ],
        }
    }

    def __init__(self, order_status=201):
        self.calls = []
        self.order_status = order_status

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(("GET", url, params))
        if url.endswith("/accountNumbers"):
            return _FakeResponse(self.ACCOUNTS)
        if url.endswith("/HASH123"):
            return _FakeResponse(self.ACCOUNT_DETAIL)
        return _FakeResponse({}, status_code=404)

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(("POST", url, json))
        return _FakeResponse(
            None, status_code=self.order_status,
            headers={"Location": "https://x/trader/v1/accounts/HASH123/orders/999"},
        )

    def delete(self, url, headers=None, timeout=None):
        self.calls.append(("DELETE", url, None))
        return _FakeResponse(None, status_code=200)


def test_account_hash_resolves(authed_cfg):
    session = _FakeSession()
    b = SchwabBroker(cfg=authed_cfg, sandbox=True, base_url="https://x", session=session)
    assert b.account_hash() == "HASH123"


def test_get_account_parses_positions(authed_cfg):
    session = _FakeSession()
    b = SchwabBroker(cfg=authed_cfg, sandbox=True, base_url="https://x", session=session)
    acct = b.get_account()
    assert acct.cash == 50_000.0
    assert acct.positions["XLK"].qty == 100
    assert acct.positions["XLF"].qty == -40        # short = long - short qty


def test_place_order_payload_and_id(authed_cfg):
    session = _FakeSession()
    b = SchwabBroker(cfg=authed_cfg, sandbox=True, base_url="https://x", session=session)
    oid = b.place_order(Order("XLK", Side.SELL_SHORT, 12))
    assert oid == "999"                            # parsed from Location header
    post = [c for c in session.calls if c[0] == "POST"][0]
    payload = post[2]
    assert payload["orderType"] == "MARKET"
    assert payload["duration"] == "DAY"
    assert payload["orderStrategyType"] == "SINGLE"
    leg = payload["orderLegCollection"][0]
    assert leg["instruction"] == "SELL_SHORT"
    assert leg["quantity"] == 12
    assert leg["instrument"] == {"symbol": "XLK", "assetType": "EQUITY"}


def test_limit_order_includes_price(authed_cfg):
    session = _FakeSession()
    b = SchwabBroker(cfg=authed_cfg, sandbox=True, base_url="https://x", session=session)
    b.place_order(Order("XLK", Side.BUY, 5, type=OrderType.LIMIT, limit_price=199.5))
    payload = [c for c in session.calls if c[0] == "POST"][0][2]
    assert payload["price"] == 199.5


def test_live_guard_blocks_without_optin(authed_cfg, monkeypatch):
    monkeypatch.delenv("CRISP_ALLOW_LIVE", raising=False)
    session = _FakeSession()
    b = SchwabBroker(cfg=authed_cfg, sandbox=False, allow_live=True,
                     base_url="https://x", session=session)
    with pytest.raises(RuntimeError, match="live order"):
        b.place_order(Order("XLK", Side.BUY, 1))
    assert not any(c[0] == "POST" for c in session.calls)   # nothing sent


def test_live_allowed_with_full_optin(authed_cfg, monkeypatch):
    monkeypatch.setenv("CRISP_ALLOW_LIVE", "1")
    session = _FakeSession()
    b = SchwabBroker(cfg=authed_cfg, sandbox=False, allow_live=True,
                     base_url="https://x", session=session)
    assert b.place_order(Order("XLK", Side.BUY, 1)) == "999"


def test_dry_run_does_not_post(authed_cfg):
    session = _FakeSession()
    b = SchwabBroker(cfg=authed_cfg, sandbox=True, dry_run=True,
                     base_url="https://x", session=session)
    assert b.place_order(Order("XLK", Side.BUY, 1)) == "dry-run"
    assert not any(c[0] == "POST" for c in session.calls)


def test_order_http_error_raises(authed_cfg):
    session = _FakeSession(order_status=400)
    b = SchwabBroker(cfg=authed_cfg, sandbox=True, base_url="https://x", session=session)
    with pytest.raises(RuntimeError):
        b.place_order(Order("XLK", Side.BUY, 1))
