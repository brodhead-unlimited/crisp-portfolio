#!/usr/bin/env python
"""Reproduce the paper's synthetic Monte-Carlo comparison.

Reports each method's out-of-sample Sharpe as a fraction of the oracle Sharpe,
averaged over many seeds, for both oracle and noisy signals.

    python scripts/synthetic_study.py --signal oracle
    python scripts/synthetic_study.py --signal noisy --seeds 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crispfolio.synthetic import REGIMES, TrialConfig, run_study


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--signal", choices=["oracle", "mu_oracle", "noisy"], default="oracle")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--snr", type=float, default=0.10)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--cov", choices=["lw", "sample"], default="lw")
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    cfg = TrialConfig(n=args.n, snr=args.snr, signal=args.signal,
                      cov_estimator=args.cov, gamma=args.gamma)
    t_grid = [0.6, 1.0, 2.0, 5.0]

    print(f"Synthetic study: signal={args.signal}, N={args.n}, "
          f"gamma={args.gamma}, seeds={args.seeds}, cov={args.cov}")
    df = run_study(list(REGIMES), t_grid, cfg, n_seeds=args.seeds)

    # Mean across regimes per (method, T/N), then a method ranking.
    pivot = (df.pivot_table(index="method", columns="T/N", values="frac_oracle")
               .round(3))
    pivot["mean"] = pivot.mean(axis=1).round(3)
    pivot = pivot.sort_values("mean", ascending=False)
    print("\n=== Fraction of oracle Sharpe (avg over regimes) ===")
    print(pivot.to_string())

    print("\n=== By regime (T/N=1.0) ===")
    sub = df[df["T/N"] == 1.0].pivot_table(index="method", columns="regime",
                                           values="frac_oracle").round(3)
    print(sub.sort_values("factor", ascending=False).to_string())

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"synthetic_{args.signal}.csv", index=False)
    print(f"\nWrote {outdir}/synthetic_{args.signal}.csv")


if __name__ == "__main__":
    main()
