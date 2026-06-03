#!/usr/bin/env python
"""Advance the live paper portfolio by one trading day and persist the ledger.

Intended to run on a daily schedule (e.g. a GitHub Action). It loads the JSON
ledger, pulls recent prices, marks to market / rebalances on cadence, writes
the ledger back, and also emits a compact ``paper.json`` for the website.

    python scripts/paper_step.py --source yfinance

Defaults to yfinance (free, ~15-min delayed) because it runs unattended:
Schwab's OAuth refresh token expires every 7 days and needs an interactive
browser login, so it can't sustain a scheduled job.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
    raise ValueError(f"unknown source: {name!r}")


def web_payload(ledger: Ledger) -> dict:
    """Compact, public-safe snapshot for the website."""
    last_prices_value = ledger.history_value[-1] if ledger.history_value else None
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
        "value": last_prices_value,
        "rebalances": ledger.rebalance_count,
        "last_rebalance": ledger.last_rebalance,
        "cadence_days": ledger.rebalance_every,
        "holdings": holdings,
        "equity": {
            "dates": ledger.history_dates,
            "series": ledger.history_equity,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["yfinance", "schwab"], default="yfinance")
    p.add_argument("--ledger", default="paper/ledger.json",
                   help="path to the persistent JSON ledger")
    p.add_argument("--web-out", default="paper/paper.json",
                   help="compact snapshot for the website")
    p.add_argument("--strategy", default="crisp")
    p.add_argument("--capital", type=float, default=100_000.0)
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

    source = make_source(args.source)
    acted = step(ledger, source, as_of=args.as_of)

    if acted:
        ledger.save(ledger_path)
        Path(args.web_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.web_out).write_text(json.dumps(web_payload(ledger), indent=2))
        eq = ledger.history_equity[-1]
        print(f"Stepped to {ledger.last_date}: value=${ledger.history_value[-1]:,.2f} "
              f"(growth x{eq:.4f}); {ledger.rebalance_count} rebalances total.")
    else:
        print(f"No new close since {ledger.last_date}; nothing to do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
