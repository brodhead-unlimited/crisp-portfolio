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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crispfolio.broker import AlpacaBroker, PaperBroker, SchwabBroker
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
    if name == "alpaca":
        return AlpacaBroker(paper=True, allow_live=allow_live, dry_run=dry_run)
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


def alpaca_web_payload(broker, ledger: Ledger, prev: dict | None = None) -> dict:
    """Website snapshot sourced from Alpaca itself.

    Equity + curve come from Alpaca (its values are authoritative and intraday),
    not the engine's per-step mark. Schema matches what ``LivePaper.tsx`` reads,
    so the site needs no change.
    """
    value = broker.equity()

    base = ledger.initial_capital or 100_000.0

    def points(fmt: str, **kwargs) -> list[tuple[str, float]]:
        hist = broker.get_portfolio_history(**kwargs)
        out: list[tuple[str, float]] = []
        for t, e in zip(hist["timestamp"], hist["equity"]):
            # Skip non-positive points: Alpaca's portfolio history reports 0 (or a
            # tiny negative) for timestamps in the window that predate the account's
            # existence/funding. Left in, those zeros drag the chart's y-axis to 0
            # and flatten the real ~1.0 curve.
            if e is None or e <= 0:
                continue
            out.append((datetime.fromtimestamp(t, tz=timezone.utc).strftime(fmt), round(e / base, 6)))
        return out

    # Hourly curve for the longest window Alpaca allows (intraday timeframes are
    # capped at 30-day queries). The published curve keeps hourly cadence for the
    # portfolio's WHOLE life: each run re-fetches the recent window and keeps the
    # previously published points that fall before it, so nothing ages out.
    start_day = date.today() - timedelta(days=29)
    if ledger.inception:
        start_day = max(start_day, date.fromisoformat(ledger.inception))
    hourly = points("%Y-%m-%d %H:%M", start=f"{start_day.isoformat()}T00:00:00Z",
                    period=None, timeframe="1H")
    merged = hourly
    if prev and prev.get("equity", {}).get("dates"):
        prev_pts = list(zip(prev["equity"]["dates"], prev["equity"]["series"]))
        if hourly:
            # Drop carried points on any DAY the fresh window covers, so old
            # daily-cadence points can't linger alongside that day's hourly ones.
            first_day = hourly[0][0][:10]
            merged = [p for p in prev_pts if p[0][:10] < first_day] + hourly
        else:
            merged = prev_pts
    dates = [d for d, _ in merged]
    series = [v for _, v in merged]

    holdings = sorted(
        ({"ticker": t, "weight": round(w, 4)} for t, w in ledger.weights.items()),
        key=lambda h: h["weight"],
        reverse=True,
    )
    return {
        "strategy": ledger.strategy,
        "inception": ledger.inception,
        "as_of": dates[-1] if dates else ledger.last_date,
        "initial_capital": ledger.initial_capital,
        "value": round(value, 2),
        "rebalances": ledger.rebalance_count,
        "last_rebalance": ledger.last_rebalance,
        "cadence_days": ledger.rebalance_every,
        "holdings": holdings,
        "last_orders": ledger.last_orders,
        "equity": {"dates": dates, "series": series},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--broker", choices=["paper", "alpaca", "schwab"], default="paper")
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

    # A dry-run previews orders but must never persist (it would advance the
    # ledger's last_date and make the next real run a no-op).
    if args.dry_run:
        print(f"[dry-run] acted={acted}; {len(ledger.last_orders)} order(s) computed; "
              "nothing placed or written.")
        return 0

    Path(args.web_out).parent.mkdir(parents=True, exist_ok=True)

    if args.broker == "alpaca":
        # Refresh the site snapshot from Alpaca every run (hourly intraday),
        # whether or not a rebalance happened this step. The previous snapshot
        # carries the hourly history older than Alpaca's 30-day intraday window.
        prev = None
        web_path = Path(args.web_out)
        if web_path.exists():
            try:
                prev = json.loads(web_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                print(f"warning: could not read previous snapshot ({exc})")
        payload = alpaca_web_payload(broker, ledger, prev=prev)
        Path(args.web_out).write_text(json.dumps(payload, indent=2))
        if acted:
            ledger.save(ledger_path)
        print(f"Alpaca snapshot: equity=${payload['value']:,.2f}, "
              f"{len(payload['equity']['series'])} curve point(s); "
              f"{'rebalanced' if acted else 'no new close'}.")
        return 0

    # paper / schwab: only write when a new close advanced the book
    if acted:
        ledger.save(ledger_path)
        Path(args.web_out).write_text(json.dumps(web_payload(ledger), indent=2))
        eq = ledger.history_equity[-1]
        print(f"Stepped to {ledger.last_date}: value=${ledger.history_value[-1]:,.2f} "
              f"(growth x{eq:.4f}); {ledger.rebalance_count} rebalances total, "
              f"{len(ledger.last_orders)} order(s) at last rebalance.")
    else:
        print(f"No new close since {ledger.last_date}; nothing to do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
