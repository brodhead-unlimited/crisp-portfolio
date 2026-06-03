#!/usr/bin/env python
"""Interactive Schwab OAuth login.

Run this ONCE in a real terminal (it needs a browser + stdin):

    python scripts/schwab_login.py

It prints an authorize URL, you approve in the browser, paste the redirected
URL back, and the resulting token set is cached at
``~/.config/crispfolio/schwab_token.json``.  After that, the token refreshes
itself for ~7 days; rerun this when the refresh token expires.

Credentials (App Key / Secret) are read from the macOS Keychain via
``crispfolio.config.SchwabConfig`` — store them first with:

    keyring set schwab app_key
    keyring set schwab app_secret
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests

from crispfolio.config import SchwabConfig
from crispfolio.schwab_auth import SchwabAuthError, login, get_access_token

ACCOUNTS_URL = "https://api.schwabapi.com/trader/v1/accounts/accountNumbers"


def _verify(cfg: SchwabConfig) -> bool:
    """Prove the token works by hitting a low-risk authenticated endpoint."""
    token = get_access_token(cfg)
    resp = requests.get(
        ACCOUNTS_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if resp.status_code == 200:
        n = len(resp.json()) if isinstance(resp.json(), list) else "?"
        print(f"Verified: authenticated call succeeded ({n} account(s) visible).")
        return True
    print(
        f"Token cached, but the verification call returned {resp.status_code}: "
        f"{resp.text}\n(The login may still be fine — this endpoint needs the "
        f"account read scope.)"
    )
    return False


def main() -> int:
    try:
        cfg = SchwabConfig.load()
    except RuntimeError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2

    if not sys.stdin.isatty():
        print(
            "This script needs an interactive terminal (it reads the pasted "
            "callback URL from stdin). Run it directly in Terminal.app, not via "
            "a non-interactive shell.",
            file=sys.stderr,
        )
        return 2

    try:
        login(cfg)
    except SchwabAuthError as e:
        print(f"\nLogin failed: {e}", file=sys.stderr)
        return 1

    try:
        _verify(cfg)
    except Exception as e:  # verification is best-effort
        print(f"Token cached, but verification call errored: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
