"""Schwab OAuth2 handshake — login once, then cached/refreshed tokens.

This module is the *private* (brokerage) side of the project; the public
website only ever consumes the sanitised ``results.json`` and never imports it.

Flow (three-legged OAuth, authorization-code grant):

1. Open the authorize URL in a browser, log in, approve.
2. Schwab redirects to the registered callback (``https://127.0.0.1:8182``)
   with ``?code=...``.  Nothing listens there, so the browser shows a
   connection error — that's expected; the value we need is in the URL bar.
3. Paste the full redirected URL back here.  We exchange the code for an
   access token (~30 min) plus a refresh token (~7 days) and cache both
   outside the repo (``~/.config/crispfolio/schwab_token.json``, mode 600).

After that, ``get_access_token`` silently refreshes the access token from the
refresh token until the refresh token itself expires (~7 days), at which point
``login`` must be run again.

Endpoints per Schwab's API docs:
    authorize : https://api.schwabapi.com/v1/oauth/authorize
    token     : https://api.schwabapi.com/v1/oauth/token
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from urllib.parse import urlencode, urlparse, parse_qs

import requests

from .config import SchwabConfig

AUTHORIZE_URL = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

# Schwab access tokens live 30 min; refresh ~60s early to avoid edge races.
_EXPIRY_SKEW_SECONDS = 60


class SchwabAuthError(RuntimeError):
    """Raised when the OAuth handshake or a refresh fails."""


@dataclass
class Token:
    """A cached token set plus the wall-clock time it was obtained."""

    access_token: str
    refresh_token: str
    expires_in: int          # seconds the access token is valid for (~1800)
    obtained_at: float       # unix time the token set was minted
    token_type: str = "Bearer"
    scope: str = ""

    @property
    def access_expires_at(self) -> float:
        return self.obtained_at + self.expires_in

    def access_expired(self, now: float) -> bool:
        return now >= self.access_expires_at - _EXPIRY_SKEW_SECONDS

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_in": self.expires_in,
            "obtained_at": self.obtained_at,
            "token_type": self.token_type,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Token":
        return cls(
            access_token=d["access_token"],
            refresh_token=d["refresh_token"],
            expires_in=int(d.get("expires_in", 1800)),
            obtained_at=float(d["obtained_at"]),
            token_type=d.get("token_type", "Bearer"),
            scope=d.get("scope", ""),
        )

    @classmethod
    def from_response(cls, payload: dict, now: float) -> "Token":
        try:
            return cls(
                access_token=payload["access_token"],
                refresh_token=payload["refresh_token"],
                expires_in=int(payload.get("expires_in", 1800)),
                obtained_at=now,
                token_type=payload.get("token_type", "Bearer"),
                scope=payload.get("scope", ""),
            )
        except KeyError as e:  # pragma: no cover - defensive
            raise SchwabAuthError(
                f"token response missing field {e}; got keys {list(payload)}"
            ) from e


# --------------------------------------------------------------------------- #
# token cache
# --------------------------------------------------------------------------- #
def load_token(cfg: SchwabConfig) -> Token | None:
    """Return the cached token set, or ``None`` if no cache exists."""
    if not cfg.token_path.exists():
        return None
    with cfg.token_path.open() as f:
        return Token.from_dict(json.load(f))


def save_token(cfg: SchwabConfig, token: Token) -> None:
    """Persist the token set with owner-only permissions (mode 600)."""
    cfg.token_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.token_path.open("w") as f:
        json.dump(token.to_dict(), f, indent=2)
    cfg.token_path.chmod(0o600)


# --------------------------------------------------------------------------- #
# HTTP legs
# --------------------------------------------------------------------------- #
def _basic_auth_header(cfg: SchwabConfig) -> str:
    raw = f"{cfg.app_key}:{cfg.app_secret}".encode()
    return "Basic " + base64.b64encode(raw).decode()


def authorize_url(cfg: SchwabConfig) -> str:
    """The URL the user opens in a browser to grant access."""
    query = urlencode(
        {
            "client_id": cfg.app_key,
            "redirect_uri": cfg.callback_url,
            "response_type": "code",
        }
    )
    return f"{AUTHORIZE_URL}?{query}"


def extract_code(redirected_url: str) -> str:
    """Pull the ``code`` parameter out of the pasted callback URL.

    Schwab appends ``@`` to the code; the token endpoint expects the value
    exactly as returned, so we hand back the raw query value untouched.
    """
    parsed = urlparse(redirected_url.strip())
    params = parse_qs(parsed.query)
    codes = params.get("code")
    if not codes:
        raise SchwabAuthError(
            "no 'code' parameter found in the pasted URL — make sure you copied "
            "the full address from the browser after approving access."
        )
    return codes[0]


def _post_token(cfg: SchwabConfig, data: dict, *, now: float) -> Token:
    resp = requests.post(
        TOKEN_URL,
        headers={
            "Authorization": _basic_auth_header(cfg),
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data=data,
        timeout=30,
    )
    if resp.status_code != 200:
        raise SchwabAuthError(
            f"token endpoint returned {resp.status_code}: {resp.text}"
        )
    return Token.from_response(resp.json(), now)


def exchange_code(cfg: SchwabConfig, code: str, *, now: float) -> Token:
    """Trade an authorization code for an access + refresh token set."""
    return _post_token(
        cfg,
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": cfg.callback_url,
        },
        now=now,
    )


def refresh_token(cfg: SchwabConfig, token: Token, *, now: float) -> Token:
    """Mint a fresh access token from the stored refresh token."""
    return _post_token(
        cfg,
        {
            "grant_type": "refresh_token",
            "refresh_token": token.refresh_token,
        },
        now=now,
    )


# --------------------------------------------------------------------------- #
# high-level entry points
# --------------------------------------------------------------------------- #
def get_access_token(cfg: SchwabConfig, *, now: float | None = None) -> str:
    """Return a currently-valid access token, refreshing if needed.

    Raises ``SchwabAuthError`` if there is no cached token (run ``login``) or
    if the refresh token has expired (run ``login`` again).
    """
    now = time.time() if now is None else now
    token = load_token(cfg)
    if token is None:
        raise SchwabAuthError(
            f"no cached Schwab token at {cfg.token_path}; run the login flow "
            f"(python scripts/schwab_login.py) first."
        )
    if token.access_expired(now):
        try:
            token = refresh_token(cfg, token, now=now)
        except SchwabAuthError as e:
            raise SchwabAuthError(
                "could not refresh the access token (the 7-day refresh token "
                f"has likely expired) — run the login flow again. Cause: {e}"
            ) from e
        save_token(cfg, token)
    return token.access_token


def login(
    cfg: SchwabConfig,
    *,
    open_browser: bool = True,
    now: float | None = None,
) -> Token:
    """Run the interactive three-legged OAuth handshake and cache the result.

    Requires a real terminal (it reads the pasted callback URL from stdin).
    """
    now = time.time() if now is None else now
    url = authorize_url(cfg)

    print("\n1. Open this URL in your browser and approve access:\n")
    print(f"   {url}\n")
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass  # headless / no browser — the printed URL still works

    print(
        "2. After approving, your browser will try to load "
        f"{cfg.callback_url} and show a connection error. That's expected.\n"
        "   Copy the FULL address from the browser's URL bar and paste it below.\n"
    )
    redirected = input("Paste the full redirected URL here: ").strip()
    code = extract_code(redirected)

    token = exchange_code(cfg, code, now=now)
    save_token(cfg, token)
    print(f"\nSuccess — token cached at {cfg.token_path} (mode 600).")
    return token
