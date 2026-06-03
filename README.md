# crispfolio

Implementation and backtesting of the portfolio-construction methods from
Wuebben (2026), *"Beyond De Prado and Cotton: Hierarchical and Iterative
Methods for General Mean-Variance Portfolios"* ([arXiv:2604.23833]), together
with the classical baselines they are compared against.

## Methods

| Method | Signal-aware? | Idea |
|---|---|---|
| `equal_weight` (1/N) | no | naive diversification |
| `min_variance` | no | global minimum-variance, Σ⁻¹1 |
| `markowitz` | yes | unconstrained Σ⁻¹μ |
| `hrp_weights` | no | Hierarchical Risk Parity (de Prado 2016) |
| `schur_weights` | no | Cotton (2024) Schur-complement allocation, γ∈[0,1] |
| `hrp_mu_weights` | yes | HRP with signed inverse-variance reps + 2×2 mean-variance splits |
| `hrp_sigma_mu_weights` | yes | HRP with recursive local mean-variance reps (L¹-normalised) |
| `crisp_weights` | yes | **CRISP**: solve `((1-γ)D + γΣ) w = μ` by Gauss–Seidel |

**CRISP** is the centrepiece: it preserves variances exactly and shrinks only
the off-diagonal correlations by γ, interpolating between a diagonal rule
(γ=0) and full Markowitz (γ=1). It is solved with scalar Gauss–Seidel sweeps
that converge for any SPD Σ.

## Layout

```
src/crispfolio/
  data/           DataSource interface + yfinance adapter (Schwab can drop in)
  estimators.py   mu and Sigma estimators (sample, Ledoit-Wolf)
  methods/        the allocators above
  backtest.py     walk-forward engine with turnover costs
  metrics.py      Sharpe, vol, drawdown, turnover
  strategies.py   estimator+allocator factories for the backtester
scripts/
  run_backtest.py walk-forward backtest on public data -> results/
tests/            correctness checks (CRISP convergence, endpoints, etc.)
```

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e . pytest
pytest -q
python scripts/run_backtest.py --start 2010-01-01 --cost-bps 5
```

Outputs land in `results/`: `summary.csv`, `equity_curves.csv`,
`results.json` (compact, for web display), and `equity_curves.png`.

## Experiments

```bash
# 1. Real-data walk-forward backtest of the full suite
python scripts/run_backtest.py --start 2010-01-01 --cost-bps 5

# 2. Synthetic Monte-Carlo reproduction (fraction of oracle Sharpe)
python scripts/synthetic_study.py --signal oracle       # both moments known
python scripts/synthetic_study.py --signal mu_oracle     # known mu, estimated Sigma
python scripts/synthetic_study.py --signal noisy         # both estimated

# 3. Gamma sweep -> the [0.3, 0.7] sweet spot
python scripts/gamma_sweep.py                            # synthetic
python scripts/gamma_sweep.py --real --start 2010-01-01  # real backtest

# 4. Does a real signal (momentum) help the signal-aware methods?
python scripts/signal_comparison.py --start 2010-01-01
```

### What we found (reproducing the paper)

- **Oracle signals:** Markowitz = 100% of oracle Sharpe, **CRISP ≈ 96%**; the
  signal-blind methods (HRP, MinVar, Cotton) collapse to ~0% because they
  ignore μ; HRP-μ / HRP-Σμ land at ~75–82%.
- **Known μ, estimated Σ:** **CRISP delivers 84–93% of oracle** across
  T/N ∈ [0.6, 5] and **beats raw Markowitz at every T/N**, by the widest
  margin when data is scarce — the paper's core claim.
- **γ sweep:** CRISP peaks at **γ ≈ 0.6**, with the whole **[0.3, 0.7] band
  beating both γ=0 (diagonal) and γ=1 (full Markowitz)** — intermediate
  correlation shrinkage strictly dominates.
- **Real data + signals:** the value of the signal-aware methods tracks signal
  quality — switching μ from sample-mean to 12-1 momentum lifts CRISP, HRP-μ,
  HRP-Σμ and Markowitz Sharpe across the board.

## Data

Research/backtesting uses free public data via `yfinance` behind a
`DataSource` interface. A `SchwabDataSource` implements the same interface for
live prices from the Charles Schwab market-data API — swap it in for
`YFinanceSource` and nothing downstream changes.

### Schwab setup

```bash
# 1. store the App Key / Secret (Client ID / Client Secret) in the macOS Keychain
keyring set schwab app_key
keyring set schwab app_secret

# 2. one-time OAuth login (opens a browser; paste the redirected URL back)
python scripts/schwab_login.py
```

Credentials live only in the Keychain; the OAuth token is cached at
`~/.config/crispfolio/schwab_token.json` (mode 600) and auto-refreshes for ~7
days. Nothing secret is committed. Note Schwab's `pricehistory` close is
split- but **not** dividend-adjusted, unlike yfinance's `auto_adjust` — keep
yfinance for long total-return backtests; use Schwab for live/holdings.

[arXiv:2604.23833]: https://arxiv.org/abs/2604.23833
