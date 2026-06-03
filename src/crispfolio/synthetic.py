"""Synthetic Monte-Carlo study, following Wuebben (2026), Section 6.

We generate a known population (mu_true, Sigma_true) under several covariance
regimes, draw a finite sample of T returns, hand each allocator either the
*oracle* moments or noisy *sample* estimates, and score the resulting weights
by their true (population) Sharpe ratio

    S(w) = (w' mu_true) / sqrt(w' Sigma_true w),

which is scale-invariant, so weight normalisation does not affect it.  The
best attainable value is the oracle Sharpe sqrt(mu' Sigma^{-1} mu); we report
each method as a fraction of that ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import methods as m
from .estimators import ledoit_wolf_cov as _lw_array  # noqa: F401 (kept for parity)

# --------------------------------------------------------------------------- #
# Covariance regimes -> (Sigma_true, mu_true)
# --------------------------------------------------------------------------- #
def _make_mu(sigma: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    """Draw a true mean whose population Sharpe ceiling is ~ snr (annualised-ish).

    We sample a direction, scale it so sqrt(mu' Sigma^{-1} mu) == snr.
    """
    n = sigma.shape[0]
    raw = rng.standard_normal(n)
    inv = np.linalg.pinv(sigma)
    norm = np.sqrt(raw @ inv @ raw)
    return raw / norm * snr if norm > 0 else raw


def factor_regime(n: int, k: int, rng: np.random.Generator):
    B = rng.standard_normal((n, k)) * 0.1
    psi = rng.uniform(0.01, 0.05, size=n)
    sigma = B @ B.T + np.diag(psi)
    return sigma


def block_regime(n: int, n_blocks: int, rng: np.random.Generator):
    rho = 0.6
    sizes = [n // n_blocks] * n_blocks
    sizes[-1] += n - sum(sizes)
    corr = np.eye(n)
    idx = 0
    for s in sizes:
        block = np.full((s, s), rho)
        np.fill_diagonal(block, 1.0)
        corr[idx:idx + s, idx:idx + s] = block
        idx += s
    vol = rng.uniform(0.1, 0.4, size=n)
    return np.outer(vol, vol) * corr


def spiked_regime(n: int, n_spikes: int, rng: np.random.Generator):
    vals = np.ones(n)
    vals[:n_spikes] = rng.uniform(5.0, 15.0, size=n_spikes)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    base = (q * vals) @ q.T
    vol = rng.uniform(0.1, 0.3, size=n)
    d = np.sqrt(np.diag(base))
    corr = base / np.outer(d, d)
    return np.outer(vol, vol) * corr


def equicorr_regime(n: int, rng: np.random.Generator, rho: float = 0.5):
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    vol = rng.uniform(0.1, 0.4, size=n)
    return np.outer(vol, vol) * corr


REGIMES = {
    "factor": lambda n, rng: factor_regime(n, max(2, n // 5), rng),
    "block": lambda n, rng: block_regime(n, max(2, n // 5), rng),
    "spiked": lambda n, rng: spiked_regime(n, max(1, n // 10), rng),
    "equicorr": lambda n, rng: equicorr_regime(n, rng),
}


# --------------------------------------------------------------------------- #
# Allocators with a uniform (cov, mu) -> weights interface
# --------------------------------------------------------------------------- #
def allocators(gamma: float = 0.5) -> dict:
    return {
        "1/N": lambda cov, mu: m.equal_weight(cov.shape[0]),
        "MinVar": lambda cov, mu: m.min_variance(cov, normalize="net"),
        "Markowitz": lambda cov, mu: m.markowitz(cov, mu, normalize="gross"),
        "HRP": lambda cov, mu: m.hrp_weights(cov, normalize="long_only"),
        "Cotton-Schur": lambda cov, mu: m.schur_weights(cov, gamma=gamma, normalize="long_only"),
        "HRP-mu": lambda cov, mu: m.hrp_mu_weights(cov, mu, normalize="gross"),
        "HRP-Sigma-mu": lambda cov, mu: m.hrp_sigma_mu_weights(cov, mu, normalize="gross"),
        "CRISP": lambda cov, mu: m.crisp_weights(cov, mu, gamma=gamma, normalize="gross"),
    }


# --------------------------------------------------------------------------- #
# Trial / study
# --------------------------------------------------------------------------- #
def true_sharpe(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    var = float(w @ sigma @ w)
    if var <= 0:
        return 0.0
    return float((w @ mu) / np.sqrt(var))


def oracle_sharpe(mu: np.ndarray, sigma: np.ndarray) -> float:
    inv = np.linalg.pinv(sigma)
    return float(np.sqrt(max(mu @ inv @ mu, 0.0)))


@dataclass
class TrialConfig:
    n: int = 30
    snr: float = 0.10          # population Sharpe ceiling
    signal: str = "noisy"      # 'oracle' or 'noisy'
    cov_estimator: str = "lw"  # 'lw' or 'sample'
    gamma: float = 0.5


def _estimate(sample: np.ndarray, how: str):
    import pandas as pd

    df = pd.DataFrame(sample)
    if how == "lw":
        from .estimators import ledoit_wolf_cov
        return ledoit_wolf_cov(df)
    from .estimators import sample_cov
    return sample_cov(df)


def run_trial(regime: str, t_over_n: float, cfg: TrialConfig, seed: int) -> dict:
    """One draw: returns {method: fraction_of_oracle_sharpe}."""
    rng = np.random.default_rng(seed)
    sigma = REGIMES[regime](cfg.n, rng)
    mu = _make_mu(sigma, cfg.snr, rng)

    T = max(cfg.n + 2, int(round(t_over_n * cfg.n)))
    sample = rng.multivariate_normal(mu, sigma, size=T)

    if cfg.signal == "oracle":          # both moments known
        cov_in, mu_in = sigma, mu
    elif cfg.signal == "mu_oracle":     # signal known, covariance estimated
        cov_in, mu_in = _estimate(sample, cfg.cov_estimator), mu
    else:                               # both estimated from the sample
        cov_in = _estimate(sample, cfg.cov_estimator)
        mu_in = sample.mean(axis=0)

    ceiling = oracle_sharpe(mu, sigma)
    out = {}
    for name, fn in allocators(cfg.gamma).items():
        try:
            w = np.asarray(fn(cov_in, mu_in), dtype=float)
            s = true_sharpe(w, mu, sigma)
        except Exception:
            s = np.nan
        out[name] = s / ceiling if ceiling > 0 else np.nan
    return out


def run_study(regimes, t_over_n_grid, cfg: TrialConfig, n_seeds: int = 50):
    """Average fraction-of-oracle Sharpe over seeds, per (regime, T/N, method)."""
    import pandas as pd

    rows = []
    for regime in regimes:
        for ton in t_over_n_grid:
            acc: dict[str, list] = {}
            for seed in range(n_seeds):
                res = run_trial(regime, ton, cfg, seed)
                for k, v in res.items():
                    acc.setdefault(k, []).append(v)
            for method, vals in acc.items():
                rows.append({
                    "regime": regime,
                    "T/N": ton,
                    "method": method,
                    "frac_oracle": float(np.nanmean(vals)),
                    "frac_oracle_std": float(np.nanstd(vals)),
                })
    return pd.DataFrame(rows)
