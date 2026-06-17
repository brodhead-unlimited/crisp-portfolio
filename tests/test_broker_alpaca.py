"""Offline tests for AlpacaBroker — the HTTP layer is faked.

Alpaca uses static API keys (no OAuth), so there is no token fixture; an
``AlpacaConfig`` is built directly and a fake session records the requests.
"""
import json

import pytest

from crispfolio.broker.alpaca import AlpacaBroker
from crispfolio.config import ALPACA_LIVE_BASE, ALPACA_PAPER_BASE, AlpacaConfig

PAPER = ALPACA_PAPER_BASE


@pytest.fixture
def cfg():
    return AlpacaConfig(api_key="KEY", api_secret="SECRET", base_url=PAPER)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        return self._payload


class _FakeSession:
    ACCOUNT = {"cash": "50000.00", "equity": "100000.00"}
    POSITIONS = [
        {"symbol": "XLK", "qty": "100", "side": "long", "avg_entry_price": "200.0"},
        {"symbol": "XLF", "qty": "40", "side": "short", "avg_entry_price": "35.0"},
    ]
    HISTORY = {"timestamp": [1, 2, 3], "equity": [100000.0, 100500.0, 99800.0]}

    def __init__(self, order_status=200):
        self.calls = []
        self.order_status = order_status

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(("GET", url, params))
        if url.endswith("/v2/account"):
            return _FakeResponse(self.ACCOUNT)
        if url.endswith("/v2/positions"):
            return _FakeResponse(self.POSITIONS)
        if url.endswith("/portfolio/history"):
            return _FakeResponse(self.HISTORY)
        return _FakeResponse({}, status_code=404)

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(("POST", url, json))
        return _FakeResponse({"id": "ord_123"}, status_code=self.order_status)

    def delete(self, url, headers=None, timeout=None):
        self.calls.append(("DELETE", url, None))
        return _FakeResponse(None, status_code=204)


def _broker(cfg, session, base_url=PAPER, **kw):
    return AlpacaBroker(cfg=cfg, base_url=base_url, session=session, **kw)


def test_get_account_parses_long_and_short(cfg):
    b = _broker(cfg, _FakeSession())
    acct = b.get_account()
    assert acct.cash == 50000.0
    assert acct.positions["XLK"].qty == 100
    assert acct.positions["XLF"].qty == -40        # side=short -> negative


def test_place_order_payload_and_id(cfg):
    session = _FakeSession()
    b = _broker(cfg, session)
    from crispfolio.broker.base import Order, Side
    oid = b.place_order(Order("XLK", Side.SELL_SHORT, 12))
    assert oid == "ord_123"
    payload = [c for c in session.calls if c[0] == "POST"][0][2]
    assert payload == {
        "symbol": "XLK", "qty": "12", "side": "sell",
        "type": "market", "time_in_force": "day",
    }


def test_buy_to_cover_maps_to_buy(cfg):
    from crispfolio.broker.base import Order, Side
    session = _FakeSession()
    b = _broker(cfg, session)
    b.place_order(Order("XLF", Side.BUY_TO_COVER, 5))
    payload = [c for c in session.calls if c[0] == "POST"][0][2]
    assert payload["side"] == "buy"


def test_limit_order_includes_price(cfg):
    from crispfolio.broker.base import Order, OrderType, Side
    session = _FakeSession()
    b = _broker(cfg, session)
    b.place_order(Order("XLK", Side.BUY, 5, type=OrderType.LIMIT, limit_price=199.5))
    payload = [c for c in session.calls if c[0] == "POST"][0][2]
    assert payload["type"] == "limit"
    assert payload["limit_price"] == "199.5"


def test_live_guard_blocks_without_optin(cfg, monkeypatch):
    from crispfolio.broker.base import Order, Side
    monkeypatch.delenv("CRISP_ALLOW_LIVE", raising=False)
    session = _FakeSession()
    b = _broker(cfg, session, base_url=ALPACA_LIVE_BASE, allow_live=True)
    with pytest.raises(RuntimeError, match="LIVE Alpaca"):
        b.place_order(Order("XLK", Side.BUY, 1))
    assert not any(c[0] == "POST" for c in session.calls)


def test_live_allowed_with_full_optin(cfg, monkeypatch):
    from crispfolio.broker.base import Order, Side
    monkeypatch.setenv("CRISP_ALLOW_LIVE", "1")
    session = _FakeSession()
    b = _broker(cfg, session, base_url=ALPACA_LIVE_BASE, allow_live=True)
    assert b.place_order(Order("XLK", Side.BUY, 1)) == "ord_123"


def test_paper_writes_freely(cfg):
    from crispfolio.broker.base import Order, Side
    session = _FakeSession()
    b = _broker(cfg, session)                       # paper base
    assert b.place_order(Order("XLK", Side.BUY, 1)) == "ord_123"


def test_dry_run_does_not_post(cfg):
    from crispfolio.broker.base import Order, Side
    session = _FakeSession()
    b = _broker(cfg, session, dry_run=True)
    assert b.place_order(Order("XLK", Side.BUY, 1)) == "dry-run"
    assert not any(c[0] == "POST" for c in session.calls)


def test_order_http_error_raises(cfg):
    from crispfolio.broker.base import Order, Side
    session = _FakeSession(order_status=422)
    b = _broker(cfg, session)
    with pytest.raises(RuntimeError):
        b.place_order(Order("XLK", Side.BUY, 1))


def test_portfolio_history_parsed(cfg):
    b = _broker(cfg, _FakeSession())
    hist = b.get_portfolio_history()
    assert hist["equity"] == [100000.0, 100500.0, 99800.0]
    assert len(hist["timestamp"]) == 3
