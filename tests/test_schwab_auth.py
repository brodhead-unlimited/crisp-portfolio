"""Offline tests for the Schwab OAuth helpers.

No network: the token endpoint is monkeypatched, so these cover URL building,
code extraction, token (de)serialisation, expiry logic, and the
cache/refresh decision in ``get_access_token``.
"""
import json

import pytest

from crispfolio.config import SchwabConfig
from crispfolio import schwab_auth as auth


@pytest.fixture
def cfg(tmp_path):
    return SchwabConfig(
        app_key="APPKEY",
        app_secret="APPSECRET",
        callback_url="https://127.0.0.1:8182",
        token_path=tmp_path / "schwab_token.json",
    )


def _token_payload(access="acc", refresh="ref", expires_in=1800):
    return {
        "access_token": access,
        "refresh_token": refresh,
        "expires_in": expires_in,
        "token_type": "Bearer",
        "scope": "api",
    }


def test_authorize_url_contains_client_and_callback(cfg):
    url = auth.authorize_url(cfg)
    assert url.startswith(auth.AUTHORIZE_URL)
    assert "client_id=APPKEY" in url
    assert "response_type=code" in url
    # callback is url-encoded
    assert "redirect_uri=https%3A%2F%2F127.0.0.1%3A8182" in url


def test_extract_code_from_redirected_url():
    pasted = "https://127.0.0.1:8182/?code=ABC123%40&session=xyz"
    assert auth.extract_code(pasted) == "ABC123@"


def test_extract_code_missing_raises():
    with pytest.raises(auth.SchwabAuthError):
        auth.extract_code("https://127.0.0.1:8182/?session=xyz")


def test_token_roundtrip(cfg):
    tok = auth.Token.from_response(_token_payload(), now=1000.0)
    auth.save_token(cfg, tok)
    assert cfg.token_path.exists()
    # mode 600
    assert (cfg.token_path.stat().st_mode & 0o777) == 0o600
    loaded = auth.load_token(cfg)
    assert loaded.access_token == "acc"
    assert loaded.refresh_token == "ref"
    assert loaded.obtained_at == 1000.0


def test_access_expired_logic():
    tok = auth.Token.from_response(_token_payload(expires_in=1800), now=0.0)
    assert not tok.access_expired(now=0.0)
    assert not tok.access_expired(now=1700.0)
    # within the 60s skew window -> treated as expired
    assert tok.access_expired(now=1750.0)
    assert tok.access_expired(now=2000.0)


def test_get_access_token_no_cache_raises(cfg):
    with pytest.raises(auth.SchwabAuthError):
        auth.get_access_token(cfg, now=0.0)


def test_get_access_token_returns_cached_when_fresh(cfg, monkeypatch):
    auth.save_token(cfg, auth.Token.from_response(_token_payload(), now=0.0))

    def _boom(*a, **k):
        raise AssertionError("should not refresh a fresh token")

    monkeypatch.setattr(auth, "refresh_token", _boom)
    assert auth.get_access_token(cfg, now=100.0) == "acc"


def test_get_access_token_refreshes_when_expired(cfg, monkeypatch):
    auth.save_token(cfg, auth.Token.from_response(_token_payload(), now=0.0))

    def _fake_refresh(cfg_, token, *, now):
        return auth.Token.from_response(
            _token_payload(access="acc2", refresh="ref2"), now=now
        )

    monkeypatch.setattr(auth, "refresh_token", _fake_refresh)
    out = auth.get_access_token(cfg, now=99999.0)
    assert out == "acc2"
    # refreshed token was persisted
    assert auth.load_token(cfg).access_token == "acc2"


def test_get_access_token_refresh_failure_raises(cfg, monkeypatch):
    auth.save_token(cfg, auth.Token.from_response(_token_payload(), now=0.0))

    def _fail(cfg_, token, *, now):
        raise auth.SchwabAuthError("refresh token expired")

    monkeypatch.setattr(auth, "refresh_token", _fail)
    with pytest.raises(auth.SchwabAuthError):
        auth.get_access_token(cfg, now=99999.0)
