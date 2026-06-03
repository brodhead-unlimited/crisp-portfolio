"""Factory of backtest-ready strategies.

Each returns a callable ``f(window_returns) -> weights`` combining a moment
estimator with an allocator.  ``mu_scale`` annualises/raw-scales the sample
mean; it only affects the *direction*-preserving methods' risk appetite, not
the relative weights after gross normalisation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import estimators as est
from . import methods as m
from . import signals as sig


def equal_weight():
    def f(window: pd.DataFrame) -> np.ndarray:
        return m.equal_weight(window.shape[1])

    return f


def min_variance(cov_estimator=est.ledoit_wolf_cov):
    def f(window: pd.DataFrame) -> np.ndarray:
        return m.min_variance(cov_estimator(window))

    return f


def markowitz(cov_estimator=est.ledoit_wolf_cov, mu_fn=sig.sample_mean, normalize="gross"):
    def f(window: pd.DataFrame) -> np.ndarray:
        return m.markowitz(cov_estimator(window), mu_fn(window), normalize=normalize)

    return f


def hrp(cov_estimator=est.sample_cov):
    def f(window: pd.DataFrame) -> np.ndarray:
        return m.hrp_weights(cov_estimator(window))

    return f


def schur(gamma=0.5, cov_estimator=est.sample_cov):
    def f(window: pd.DataFrame) -> np.ndarray:
        return m.schur_weights(cov_estimator(window), gamma=gamma)

    return f


def hrp_mu(cov_estimator=est.sample_cov, mu_fn=sig.sample_mean):
    def f(window: pd.DataFrame) -> np.ndarray:
        return m.hrp_mu_weights(cov_estimator(window), mu_fn(window))

    return f


def hrp_sigma_mu(cov_estimator=est.sample_cov, mu_fn=sig.sample_mean):
    def f(window: pd.DataFrame) -> np.ndarray:
        return m.hrp_sigma_mu_weights(cov_estimator(window), mu_fn(window))

    return f


def crisp(gamma=0.5, sweeps=100, cov_estimator=est.ledoit_wolf_cov,
          mu_fn=sig.sample_mean, normalize="gross"):
    def f(window: pd.DataFrame) -> np.ndarray:
        cov = cov_estimator(window)
        mu = mu_fn(window)
        return m.crisp_weights(cov, mu, gamma=gamma, max_sweeps=sweeps, normalize=normalize)

    return f


def default_suite() -> dict:
    """A representative comparison set spanning all method families."""
    return {
        "1/N": equal_weight(),
        "MinVar (LW)": min_variance(),
        "Markowitz (LW)": markowitz(),
        "HRP": hrp(),
        "Cotton-Schur (g=0.5)": schur(gamma=0.5),
        "HRP-mu": hrp_mu(),
        "HRP-Sigma-mu": hrp_sigma_mu(),
        "CRISP (g=0.5)": crisp(gamma=0.5),
    }
