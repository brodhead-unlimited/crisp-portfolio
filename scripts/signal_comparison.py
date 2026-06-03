#!/usr/bin/env python
"""Does a real alpha signal help the signal-aware methods on real data?

Backtests CRISP, HRP-mu, HRP-Sigma-mu and Markowitz with two mu sources --
the naive sample mean vs 12-1 momentum -- alongside the signal-blind anchors
(1/N, HRP). The hypothesis (and the paper's premise) is that the signal-aware
methods only earn their keep when fed a decent signal.

    python scripts/signal_comparison.py --start 2010-01-01 --cost-bps 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crispfolio import signals as sig
from crispfolio import strategies as S
from crispfolio.backtest import run_backtest
from crispfolio.data import YFinanceSource

DEFAULT_TICKERS = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
    "TLT", "IEF", "GLD", "EFA", "EEM", "VNQ",
]


def build_suite(gamma: float) -> dict:
    suite = {
        "1/N": S.equal_weight(),
        "HRP": S.hrp(),
        "MinVar (LW)": S.min_variance(),
    }
    for sig_name, mu_fn in [("mean", sig.sample_mean), ("mom", sig.momentum)]:
        suite[f"CRISP[{sig_name}]"] = S.crisp(gamma=gamma, mu_fn=mu_fn)
        suite[f"HRP-mu[{sig_name}]"] = S.hrp_mu(mu_fn=mu_fn)
        suite[f"HRP-Smu[{sig_name}]"] = S.hrp_sigma_mu(mu_fn=mu_fn)
        suite[f"Markowitz[{sig_name}]"] = S.markowitz(mu_fn=mu_fn)
    return suite


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--rebalance-every", type=int, default=21)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    data = YFinanceSource().get_prices(DEFAULT_TICKERS, args.start, args.end)
    rets = data.returns("simple").dropna(how="any")
    print(f"{rets.shape[0]} days x {rets.shape[1]} assets "
          f"({rets.index[0].date()} -> {rets.index[-1].date()})")

    stats = {}
    for name, strat in build_suite(args.gamma).items():
        res = run_backtest(rets, strat, name=name, lookback=args.lookback,
                           rebalance_every=args.rebalance_every, cost_bps=args.cost_bps)
        stats[name] = res.stats

    table = pd.DataFrame(stats).T[["ann_return", "ann_vol", "sharpe",
                                   "max_drawdown", "avg_turnover"]]
    pd.set_option("display.float_format", lambda x: f"{x:7.3f}")
    print("\n=== Sample-mean vs momentum signal (sorted by Sharpe) ===")
    print(table.sort_values("sharpe", ascending=False).to_string())

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "signal_comparison.csv")
    print(f"\nWrote {outdir}/signal_comparison.csv")


if __name__ == "__main__":
    main()
