"""``AlpacaBroker`` — places orders through Alpaca's Trading API.

Implements the same :class:`Broker` interface as the other brokers, so the
rebalance code is unchanged. Unlike Schwab, Alpaca uses **static API keys**
(no OAuth, no weekly re-login), so a scheduled job can run it unattended — and
its **paper** environment is a genuine broker-side simulator (real order
lifecycle, real fills, real position tracking) at zero money risk.

Environment is the base URL, and that is also what gates writes:

* **Paper** (``https://paper-api.alpaca.markets``, the default) — synthetic
  money. Orders are accepted/filled freely.
* **Live** (``https://api.alpaca.markets``) — real money. Every write is refused
  unless ``allow_live=True`` *and* env ``CRISP_ALLOW_LIVE=1``, regardless of the
  ``paper`` flag.

Alpaca expresses direction as plain ``buy``/``sell`` and infers long vs short
from the resulting position, so our four-way :class:`Side` collapses to two.

Endpoints (Trading API):
    GET    {base}/v2/account
    GET    {base}/v2/positions
    POST   {base}/v2/orders
    GET    {base}/v2/orders/{id}
    DELETE {base}/v2/orders/{id}
    GET    {base}/v2/account/portfolio/history
"""
from __future__ import annotations

import os

import requests

from ..config import ALPACA_LIVE_BASE, AlpacaConfig
from .base import Account, Broker, Order, OrderType, Position, Side

# Our explicit instructions collapse to Alpaca's buy/sell (it derives long/short
# from the current position).
_SIDE_TO_ALPACA = {
    Side.BUY: "buy",
    Side.BUY_TO_COVER: "buy",
    Side.SELL: "sell",
    Side.SELL_SHORT: "sell",
}


class AlpacaBroker(Broker):
    def __init__(
        self,
        cfg: AlpacaConfig | None = None,
        *,
        paper: bool = True,
        base_url: str | None = None,
        allow_live: bool = False,
        dry_run: bool = False,
        session: requests.Session | None = None,
    ):
        self.cfg = cfg or AlpacaConfig.load(paper=paper)
        self.paper = paper
        self.allow_live = allow_live
        self.dry_run = dry_run
        self.base = base_url or self.cfg.base_url
        self._session = session or requests.Session()

    # -- helpers ------------------------------------------------------------ #
    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": self.cfg.api_key,
            "APCA-API-SECRET-KEY": self.cfg.api_secret,
        }

    def _is_live(self) -> bool:
        return self.base.rstrip("/") == ALPACA_LIVE_BASE

    def _check_write_allowed(self) -> None:
        """Block writes to the LIVE host unless fully opted in (URL-based)."""
        if not self._is_live():
            return
        if not (self.allow_live and os.environ.get("CRISP_ALLOW_LIVE") == "1"):
            raise RuntimeError(
                f"refusing to write to the LIVE Alpaca host ({self.base}): "
                "real-money orders require allow_live=True AND env "
                "CRISP_ALLOW_LIVE=1. Use the paper host for synthetic money."
            )

    # -- Broker interface --------------------------------------------------- #
    def get_account(self) -> Account:
        r = self._session.get(f"{self.base}/v2/account", headers=self._headers(), timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"get_account returned {r.status_code}: {r.text}")
        cash = float(r.json().get("cash", 0.0))

        rp = self._session.get(f"{self.base}/v2/positions", headers=self._headers(), timeout=30)
        if rp.status_code != 200:
            raise RuntimeError(f"get positions returned {rp.status_code}: {rp.text}")
        positions: dict[str, Position] = {}
        for p in rp.json():
            sym = p["symbol"]
            mag = abs(int(float(p["qty"])))
            qty = -mag if p.get("side") == "short" else mag
            if qty == 0:
                continue
            positions[sym] = Position(sym, qty, float(p.get("avg_entry_price", 0.0)))
        return Account(cash=cash, positions=positions)

    def place_order(self, order: Order) -> str:
        payload = self._order_payload(order)
        if self.dry_run:
            print(f"[dry-run] would POST order: {payload}")
            return "dry-run"
        self._check_write_allowed()
        r = self._session.post(
            f"{self.base}/v2/orders", headers=self._headers(), json=payload, timeout=30
        )
        if r.status_code not in (200, 201):
            raise RuntimeError(f"place_order returned {r.status_code}: {r.text}")
        return r.json().get("id", "")

    def equity(self, prices=None) -> float:
        """Alpaca's own net-liquidation value — consistent even mid-settlement."""
        r = self._session.get(f"{self.base}/v2/account", headers=self._headers(), timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"equity (account) returned {r.status_code}: {r.text}")
        return float(r.json().get("equity", 0.0))

    def get_order(self, order_id: str) -> dict:
        r = self._session.get(
            f"{self.base}/v2/orders/{order_id}", headers=self._headers(), timeout=30
        )
        if r.status_code != 200:
            raise RuntimeError(f"get_order returned {r.status_code}: {r.text}")
        return r.json()

    def cancel_order(self, order_id: str) -> None:
        self._check_write_allowed()
        r = self._session.delete(
            f"{self.base}/v2/orders/{order_id}", headers=self._headers(), timeout=30
        )
        if r.status_code not in (200, 204):
            raise RuntimeError(f"cancel_order returned {r.status_code}: {r.text}")

    # -- extras ------------------------------------------------------------- #
    def get_portfolio_history(self, period: str | None = "1M", timeframe: str = "1D",
                              start: str | None = None) -> dict:
        """Alpaca's own equity time series — feeds the website curve directly.

        Alpaca allows at most two of ``start``/``end``/``period``; pass ``start``
        (RFC3339) with ``period=None`` to get everything since a date.
        """
        params: dict[str, str] = {"timeframe": timeframe}
        if start is not None:
            params["start"] = start
        elif period is not None:
            params["period"] = period
        r = self._session.get(
            f"{self.base}/v2/account/portfolio/history",
            headers=self._headers(),
            params=params,
            timeout=30,
        )
        if r.status_code != 200:
            raise RuntimeError(f"portfolio/history returned {r.status_code}: {r.text}")
        j = r.json()
        return {"timestamp": j.get("timestamp", []), "equity": j.get("equity", [])}

    # -- order shaping ------------------------------------------------------ #
    @staticmethod
    def _order_payload(order: Order) -> dict:
        payload = {
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": _SIDE_TO_ALPACA[order.side],
            "type": order.type.value.lower(),   # MARKET -> "market"
            "time_in_force": "day",
        }
        if order.type is OrderType.LIMIT:
            if order.limit_price is None:
                raise ValueError("LIMIT order requires a limit_price")
            payload["limit_price"] = str(order.limit_price)
        return payload
