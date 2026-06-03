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
