"""``SchwabBroker`` — places real orders through Schwab's Trader API.

Implements the same :class:`Broker` interface as ``PaperBroker``, so the
rebalance code is identical; only the construction changes. Authentication
reuses :func:`crispfolio.schwab_auth.get_access_token` (the cached OAuth token);
run ``scripts/schwab_login.py`` once first.

Two environments share one codebase, differing only by base URL (per Schwab's
docs, "the same credentials are used; only the API base URL changes"):

* **Sandbox** (``sandbox=True``, the default) — synthetic accounts and money.
  Safe to exercise the full order path. Point ``SCHWAB_API_BASE`` (or the
  ``base_url`` arg) at the sandbox host shown in your Developer Portal app.
* **Production** (``sandbox=False``) — real money. Guarded hard: a live order
  also requires ``allow_live=True`` *and* env ``CRISP_ALLOW_LIVE=1``.

Endpoints (Trader API):
    GET    {base}/trader/v1/accounts/accountNumbers
    GET    {base}/trader/v1/accounts/{hash}?fields=positions
    POST   {base}/trader/v1/accounts/{hash}/orders
    GET    {base}/trader/v1/accounts/{hash}/orders/{id}
    DELETE {base}/trader/v1/accounts/{hash}/orders/{id}
"""
from __future__ import annotations

import os

import requests

from ..config import SchwabConfig
from ..schwab_auth import get_access_token
from .base import Account, Broker, Order, OrderType, Position, Side

PROD_BASE = "https://api.schwabapi.com"


class SchwabBroker(Broker):
    def __init__(
        self,
        cfg: SchwabConfig | None = None,
        *,
        sandbox: bool = True,
        base_url: str | None = None,
        allow_live: bool = False,
        dry_run: bool = False,
        session: requests.Session | None = None,
    ):
        self.cfg = cfg or SchwabConfig.load()
        self.sandbox = sandbox
        self.allow_live = allow_live
        self.dry_run = dry_run
        # Sandbox host comes from the Developer Portal; configure it explicitly
        # rather than guessing. Falls back to production for live use.
        self.base = base_url or os.environ.get("SCHWAB_API_BASE") or PROD_BASE
        self._session = session or requests.Session()
        self._hash: str | None = None

    # -- helpers ------------------------------------------------------------ #
    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {get_access_token(self.cfg)}"}

    def _check_live_allowed(self) -> None:
        """Block any real-money write unless explicitly, doubly opted in."""
        if self.sandbox:
            return
        if not (self.allow_live and os.environ.get("CRISP_ALLOW_LIVE") == "1"):
            raise RuntimeError(
                "refusing to place a live order: sandbox=False requires both "
                "allow_live=True and env CRISP_ALLOW_LIVE=1."
            )

    def account_hash(self) -> str:
        """Resolve (and cache) the hashed account id used in order URLs.

        Picks the account matching ``SCHWAB_ACCOUNT`` if set, else the first.
        """
        if self._hash is not None:
            return self._hash
        url = f"{self.base}/trader/v1/accounts/accountNumbers"
        resp = self._session.get(url, headers=self._auth_headers(), timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"accountNumbers returned {resp.status_code}: {resp.text}"
            )
        accounts = resp.json()
        if not accounts:
            raise RuntimeError("Schwab returned no accounts for this login.")
        wanted = os.environ.get("SCHWAB_ACCOUNT")
        if wanted:
            for a in accounts:
                if a.get("accountNumber") == wanted:
                    self._hash = a["hashValue"]
                    break
            else:
                raise RuntimeError(
                    f"account {wanted} not visible to this login "
                    f"({len(accounts)} account(s) available)."
                )
        else:
            self._hash = accounts[0]["hashValue"]
        return self._hash

    # -- Broker interface --------------------------------------------------- #
    def get_account(self) -> Account:
        url = f"{self.base}/trader/v1/accounts/{self.account_hash()}"
        resp = self._session.get(
            url, headers=self._auth_headers(),
            params={"fields": "positions"}, timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"get_account returned {resp.status_code}: {resp.text}")
        sec = resp.json().get("securitiesAccount", {})
        balances = sec.get("currentBalances", {})
        cash = float(balances.get("cashBalance", balances.get("cashAvailableForTrading", 0.0)))
        positions: dict[str, Position] = {}
        for p in sec.get("positions", []):
            sym = p["instrument"]["symbol"]
            qty = int(float(p.get("longQuantity", 0.0)) - float(p.get("shortQuantity", 0.0)))
            if qty == 0:
                continue
            positions[sym] = Position(sym, qty, float(p.get("averagePrice", 0.0)))
        return Account(cash=cash, positions=positions)

    def place_order(self, order: Order) -> str:
        self._check_live_allowed()
        payload = self._order_payload(order)
        if self.dry_run:
            print(f"[dry-run] would POST order: {payload}")
            return "dry-run"
        url = f"{self.base}/trader/v1/accounts/{self.account_hash()}/orders"
        resp = self._session.post(
            url,
            headers={**self._auth_headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"place_order returned {resp.status_code}: {resp.text}")
        # Schwab returns the new order id in the Location header.
        location = resp.headers.get("Location", "")
        return location.rstrip("/").rsplit("/", 1)[-1] if location else ""

    def get_order(self, order_id: str) -> dict:
        url = f"{self.base}/trader/v1/accounts/{self.account_hash()}/orders/{order_id}"
        resp = self._session.get(url, headers=self._auth_headers(), timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"get_order returned {resp.status_code}: {resp.text}")
        return resp.json()

    def cancel_order(self, order_id: str) -> None:
        self._check_live_allowed()
        url = f"{self.base}/trader/v1/accounts/{self.account_hash()}/orders/{order_id}"
        resp = self._session.delete(url, headers=self._auth_headers(), timeout=30)
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"cancel_order returned {resp.status_code}: {resp.text}")

    # -- order shaping ------------------------------------------------------ #
    @staticmethod
    def _order_payload(order: Order) -> dict:
        leg = {
            "instruction": order.side.value,
            "quantity": order.qty,
            "instrument": {"symbol": order.symbol, "assetType": "EQUITY"},
        }
        payload = {
            "orderType": order.type.value,
            "session": "NORMAL",
            "duration": order.tif,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [leg],
        }
        if order.type is OrderType.LIMIT:
            if order.limit_price is None:
                raise ValueError("LIMIT order requires a limit_price")
            payload["price"] = order.limit_price
        return payload
