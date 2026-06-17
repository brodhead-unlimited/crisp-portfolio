#!/usr/bin/env python
"""Advance the live paper portfolio by one trading day and persist the ledger.

Intended to run on a daily schedule (e.g. a GitHub Action). It loads the JSON
ledger, pulls recent prices from a data source, marks to market / rebalances on
cadence through a broker, writes the ledger back, and emits a compact
``paper.json`` for the website.

    python scripts/paper_step.py --broker paper --data-source yfinance

Data source and broker are independent:

* ``--broker paper`` simulates fills locally and stores the whole book in the
  ledger. No auth, no secrets — safe for unattended CI. This drives the website.
* ``--broker schwab --sandbox`` routes orders to Schwab's sandbox (synthetic
  money) to rehearse the real execution path. Needs a prior interactive login
  (``scripts/schwab_login.py``) and so can't run unattended (Schwab's refresh
  token expires every 7 days). Use a SEPARATE ``--ledger`` so it never touches
  the public paper book.

``--data-source`` defaults to yfinance (free, ~15-min delayed); ``--broker``
defaults to paper.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crispfolio.broker import PaperBroker, SchwabBroker
from crispfolio.paper import Ledger, step

# Same cross-asset basket as the backtest, so the live curve continues it.
DEFAULT_TICKERS = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
    "TLT", "IEF", "GLD", "EFA", "EEM", "VNQ",
]


def make_source(name: str):
    if name == "yfinance":
        from crispfolio.data import YFinanceSource

        return YFinanceSource(cache=False)
    if name == "schwab":
        from crispfolio.data import SchwabDataSource

        return SchwabDataSource(cache=False)
    raise ValueError(f"unknown data source: {name!r}")


def make_broker(name: str, ledger: Ledger, *, sandbox: bool, allow_live: bool,
                dry_run: bool):
    if name == "paper":
        return ledger.make_paper_broker()
    if name == "schwab":
        return SchwabBroker(sandbox=sandbox, allow_live=allow_live, dry_run=dry_run)
    raise ValueError(f"unknown broker: {name!r}")


def web_payload(ledger: Ledger) -> dict:
    """Compact, public-safe snapshot for the website."""
    last_value = ledger.history_value[-1] if ledger.history_value else None
    holdings = sorted(
        ({"ticker": t, "weight": round(w, 4)} for t, w in ledger.weights.items()),
        key=lambda h: h["weight"],
        reverse=True,
    )
    return {
        "strategy": ledger.strategy,
        "inception": ledger.inception,
        "as_of": ledger.last_date,
        "initial_capital": ledger.initial_capital,
        "value": last_value,
        "rebalances": ledger.rebalance_count,
        "last_rebalance": ledger.last_rebalance,
        "cadence_days": ledger.rebalance_every,
        "holdings": holdings,
        "last_orders": ledger.last_orders,
        "equity": {
            "dates": ledger.history_dates,
            "series": ledger.history_equity,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--broker", choices=["paper", "schwab"], default="paper")
    p.add_argument("--data-source", choices=["yfinance", "schwab"], default="yfinance")
    p.add_argument("--ledger", default="paper/ledger.json",
                   help="path to the persistent JSON ledger")
    p.add_argument("--web-out", default="paper/paper.json",
                   help="compact snapshot for the website")
    p.add_argument("--strategy", default="crisp")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--no-short", action="store_true",
                   help="drop the strategy's short legs (long-only book)")
    p.add_argument("--sandbox", action="store_true",
                   help="(schwab broker) route to the Schwab sandbox")
    p.add_argument("--allow-live", action="store_true",
                   help="(schwab broker) permit real-money orders; also needs CRISP_ALLOW_LIVE=1")
    p.add_argument("--dry-run", action="store_true",
                   help="(schwab broker) log orders without sending")
    p.add_argument("--as-of", default=None,
                   help="cap prices at this date (YYYY-MM-DD); for backfill/testing")
    args = p.parse_args()

    ledger_path = Path(args.ledger)
    ledger = Ledger.load(ledger_path)
    if ledger is None:
        ledger = Ledger(
            strategy=args.strategy,
            tickers=DEFAULT_TICKERS,
            initial_capital=args.capital,
        )
        print(f"Initialised new {args.strategy} paper ledger.")

    source = make_source(args.data_source)
    broker = make_broker(
        args.broker, ledger,
        sandbox=args.sandbox, allow_live=args.allow_live, dry_run=args.dry_run,
    )
    acted = step(ledger, broker, source, as_of=args.as_of, allow_short=not args.no_short)

    if acted:
        ledger.save(ledger_path)
        Path(args.web_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.web_out).write_text(json.dumps(web_payload(ledger), indent=2))
        eq = ledger.history_equity[-1]
        n_orders = len(ledger.last_orders)
        print(f"Stepped to {ledger.last_date}: value=${ledger.history_value[-1]:,.2f} "
              f"(growth x{eq:.4f}); {ledger.rebalance_count} rebalances total, "
              f"{n_orders} order(s) at last rebalance.")
    else:
        print(f"No new close since {ledger.last_date}; nothing to do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
