"""Secrets and configuration for the (private) Schwab side.

Secrets are read from the macOS Keychain via ``keyring`` first, then fall back
to environment variables.  Nothing secret is ever stored in the repo: the
public website side only consumes the sanitised ``results.json`` and never
imports this module.

Store secrets once (in your own terminal, not in source):

    keyring set schwab app_key      # paste App Key  (client ID)
    keyring set schwab app_secret   # paste Secret
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_SERVICE = "schwab"

# Must match EXACTLY the callback registered in the Schwab developer portal.
DEFAULT_CALLBACK_URL = "https://127.0.0.1:8182"

# OAuth token cache lives outside the repo so it can never be committed.
DEFAULT_TOKEN_PATH = Path.home() / ".config" / "crispfolio" / "schwab_token.json"


def _get_secret(name: str) -> str | None:
    """Keychain first, then SCHWAB_<NAME> env var."""
    try:
        import keyring

        val = keyring.get_password(_SERVICE, name)
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(f"SCHWAB_{name.upper()}")


@dataclass
class SchwabConfig:
    app_key: str
    app_secret: str
    callback_url: str = DEFAULT_CALLBACK_URL
    token_path: Path = DEFAULT_TOKEN_PATH

    @classmethod
    def load(cls) -> "SchwabConfig":
        app_key = _get_secret("app_key")
        app_secret = _get_secret("app_secret")
        missing = [n for n, v in (("app_key", app_key), ("app_secret", app_secret)) if not v]
        if missing:
            raise RuntimeError(
                f"Missing Schwab secret(s): {', '.join(missing)}. "
                f"Store them with:  keyring set schwab {missing[0]}"
            )
        token_path = Path(os.environ.get("SCHWAB_TOKEN_PATH", DEFAULT_TOKEN_PATH))
        token_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(
            app_key=app_key,
            app_secret=app_secret,
            callback_url=os.environ.get("SCHWAB_CALLBACK_URL", DEFAULT_CALLBACK_URL),
            token_path=token_path,
        )


# --------------------------------------------------------------------------- #
# Alpaca — paper/live trading via static API keys (no OAuth, no token expiry)
# --------------------------------------------------------------------------- #
_ALPACA_SERVICE = "alpaca"
ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets"
ALPACA_LIVE_BASE = "https://api.alpaca.markets"

# keyring username -> ordered env-var fallbacks (support both common names)
_ALPACA_ENV = {
    "api_key": ("APCA_API_KEY_ID", "ALPACA_API_KEY"),
    "api_secret": ("APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY"),
}


def _get_alpaca_secret(name: str) -> str | None:
    """Keychain (service ``alpaca``) first, then the known env-var names."""
    try:
        import keyring

        val = keyring.get_password(_ALPACA_SERVICE, name)
        if val:
            return val
    except Exception:
        pass
    for env in _ALPACA_ENV[name]:
        v = os.environ.get(env)
        if v:
            return v
    return None


@dataclass
class AlpacaConfig:
    api_key: str
    api_secret: str
    base_url: str = ALPACA_PAPER_BASE

    @classmethod
    def load(cls, *, paper: bool = True) -> "AlpacaConfig":
        api_key = _get_alpaca_secret("api_key")
        api_secret = _get_alpaca_secret("api_secret")
        missing = [n for n, v in (("api_key", api_key), ("api_secret", api_secret)) if not v]
        if missing:
            raise RuntimeError(
                f"Missing Alpaca secret(s): {', '.join(missing)}. Store them with:  "
                f"keyring set alpaca {missing[0]}"
            )
        base = os.environ.get("ALPACA_API_BASE") or (
            ALPACA_PAPER_BASE if paper else ALPACA_LIVE_BASE
        )
        return cls(api_key=api_key, api_secret=api_secret, base_url=base)
