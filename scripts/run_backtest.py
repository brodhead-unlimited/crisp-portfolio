#!/usr/bin/env python
"""Walk-forward backtest of the full method suite on public market data.

Example:
    python scripts/run_backtest.py --start 2015-01-01 --cost-bps 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from crispfolio.backtest import run_backtest
from crispfolio.data import DataSource, YFinanceSource
from crispfolio.strategies import default_suite

# A diversified cross-asset basket (equity sectors + bonds + gold + intl).
DEFAULT_TICKERS = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
    "TLT", "IEF", "GLD", "EFA", "EEM", "VNQ",
]


def make_source(name: str) -> DataSource:
    """Build the requested price source. ``schwab`` is imported lazily so the
    default yfinance path needs no Schwab creds/login present."""
    if name == "yfinance":
        return YFinanceSource()
    if name == "schwab":
        # Schwab close is split- but NOT dividend-adjusted (unlike yfinance's
        # auto_adjust); prefer yfinance for long total-return backtests.
        from crispfolio.data import SchwabDataSource

        return SchwabDataSource()
    raise ValueError(f"unknown source: {name!r} (expected 'yfinance' or 'schwab')")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--rebalance-every", type=int, default=21)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--source", choices=["yfinance", "schwab"], default="yfinance",
                   help="price data source (default: yfinance)")
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    print(f"Fetching {len(args.tickers)} tickers from {args.start} "
          f"via {args.source} ...")
    data = make_source(args.source).get_prices(args.tickers, args.start, args.end)
    rets = data.returns("simple").dropna(how="any")
    print(f"  {rets.shape[0]} days x {rets.shape[1]} assets "
          f"({rets.index[0].date()} -> {rets.index[-1].date()})")

    results = {}
    for name, strat in default_suite().items():
        res = run_backtest(
            rets, strat, name=name,
            lookback=args.lookback,
            rebalance_every=args.rebalance_every,
            cost_bps=args.cost_bps,
        )
        results[name] = res

    table = pd.DataFrame({name: r.stats for name, r in results.items()}).T
    table = table[["ann_return", "ann_vol", "sharpe", "max_drawdown", "avg_turnover"]]
    pd.set_option("display.float_format", lambda x: f"{x:7.3f}")
    print("\n=== Out-of-sample performance ===")
    print(table.sort_values("sharpe", ascending=False).to_string())

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "summary.csv")

    equity = pd.DataFrame({name: r.equity for name, r in results.items()})
    equity.to_csv(outdir / "equity_curves.csv")

    # Compact JSON for downstream display (e.g. the website).
    payload = {
        "meta": {
            "tickers": list(rets.columns),
            "start": str(rets.index[0].date()),
            "end": str(rets.index[-1].date()),
            "lookback": args.lookback,
            "rebalance_every": args.rebalance_every,
            "cost_bps": args.cost_bps,
            "source": args.source,
        },
        "stats": {name: r.stats for name, r in results.items()},
        "equity": {
            "dates": [str(d.date()) for d in equity.index],
            "series": {name: equity[name].round(5).tolist() for name in equity.columns},
        },
    }
    (outdir / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {outdir}/summary.csv, equity_curves.csv, results.json")

    # Optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = equity.plot(figsize=(11, 6), logy=True)
        ax.set_title("Growth of $1 (log scale) — out-of-sample")
        ax.set_ylabel("equity")
        plt.tight_layout()
        plt.savefig(outdir / "equity_curves.png", dpi=120)
        print(f"Wrote {outdir}/equity_curves.png")
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
