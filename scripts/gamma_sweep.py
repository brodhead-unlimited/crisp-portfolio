#!/usr/bin/env python
"""Sweep the shrinkage parameter gamma for CRISP and Cotton-Schur.

Synthetic mode (default) sweeps gamma against fraction-of-oracle Sharpe under
the 'known mu, estimated Sigma' setup -- the regime where correlation shrinkage
matters -- and locates the sweet spot the paper reports (gamma ~ 0.3-0.7).

    python scripts/gamma_sweep.py                       # synthetic
    python scripts/gamma_sweep.py --real --start 2010-01-01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# A diversified cross-asset basket (kept in sync with run_backtest.py).
DEFAULT_TICKERS = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
    "TLT", "IEF", "GLD", "EFA", "EEM", "VNQ",
]


def sweep_synthetic(gammas, regimes, t_over_n, n, snr, seeds):
    from crispfolio.synthetic import TrialConfig, run_trial

    rows = []
    for g in gammas:
        cfg = TrialConfig(n=n, snr=snr, signal="mu_oracle", gamma=g)
        crisp_vals, schur_vals = [], []
        for regime in regimes:
            for seed in range(seeds):
                res = run_trial(regime, t_over_n, cfg, seed)
                crisp_vals.append(res["CRISP"])
                schur_vals.append(res["Cotton-Schur"])
        rows.append({
            "gamma": g,
            "CRISP": float(np.nanmean(crisp_vals)),
            "Cotton-Schur": float(np.nanmean(schur_vals)),
        })
    return pd.DataFrame(rows).set_index("gamma")


def sweep_real(gammas, args):
    from crispfolio.backtest import run_backtest
    from crispfolio.data import YFinanceSource
    from crispfolio.strategies import crisp as crisp_strat, schur as schur_strat

    data = YFinanceSource().get_prices(DEFAULT_TICKERS, args.start, args.end)
    rets = data.returns("simple").dropna(how="any")
    rows = []
    for g in gammas:
        rc = run_backtest(rets, crisp_strat(gamma=g), lookback=args.lookback,
                          rebalance_every=args.rebalance_every, cost_bps=args.cost_bps)
        rs = run_backtest(rets, schur_strat(gamma=g), lookback=args.lookback,
                          rebalance_every=args.rebalance_every, cost_bps=args.cost_bps)
        rows.append({"gamma": g, "CRISP": rc.stats["sharpe"], "Cotton-Schur": rs.stats["sharpe"]})
    return pd.DataFrame(rows).set_index("gamma")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real", action="store_true", help="sweep on real backtested data")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--rebalance-every", type=int, default=21)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--snr", type=float, default=0.10)
    p.add_argument("--t-over-n", type=float, default=1.0)
    p.add_argument("--seeds", type=int, default=40)
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    gammas = np.round(np.linspace(0.0, 1.0, 11), 2)
    if args.real:
        print(f"Real gamma sweep, {args.start} -> now")
        df = sweep_real(gammas, args)
        metric = "out-of-sample Sharpe"
    else:
        print(f"Synthetic gamma sweep (known mu, estimated Sigma), "
              f"N={args.n}, T/N={args.t_over_n}, seeds={args.seeds}")
        from crispfolio.synthetic import REGIMES
        df = sweep_synthetic(gammas, list(REGIMES),
                             args.t_over_n, args.n, args.snr, args.seeds)
        metric = "fraction of oracle Sharpe"

    print(f"\n=== gamma vs {metric} ===")
    print(df.round(3).to_string())
    for col in df.columns:
        best = df[col].idxmax()
        print(f"  best {col}: gamma={best}  ({df[col].max():.3f})")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "real" if args.real else "synthetic"
    df.to_csv(outdir / f"gamma_sweep_{tag}.csv")
    print(f"\nWrote {outdir}/gamma_sweep_{tag}.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = df.plot(marker="o", figsize=(9, 5.5))
        ax.set_xlabel("gamma (correlation shrinkage)")
        ax.set_ylabel(metric)
        ax.set_title(f"Gamma sweep ({tag})")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"gamma_sweep_{tag}.png", dpi=120)
        print(f"Wrote {outdir}/gamma_sweep_{tag}.png")
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
