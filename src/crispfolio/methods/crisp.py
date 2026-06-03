"""CRISP — Correlation-Regularised Iterative Shrinkage Portfolios.

Wuebben (2026), arXiv:2604.23833, Section 5.

Idea
----
Markowitz solves  Sigma w = mu.  CRISP instead solves the *regularised* system

    P_gamma w = mu,        P_gamma = (1 - gamma) D + gamma Sigma,

where D = diag(Sigma).  Because the diagonal of P_gamma equals
(1-gamma) sigma_ii + gamma sigma_ii = sigma_ii, the *variances are preserved
exactly* and only the off-diagonal covariances are shrunk by gamma:

    gamma = 0  ->  P_0 = D            (diagonal / naive risk-scaled rule)
    gamma = 1  ->  P_1 = Sigma        (full Markowitz)

The system is solved by scalar Gauss-Seidel sweeps, which converge for every
SPD Sigma and gamma in [0, 1] (Theorem 5.2).  The convergence rate depends on
the conditioning of the *correlation* matrix, not the covariance, so volatility
dispersion does not slow it down.
"""
from __future__ import annotations

import numpy as np

from .common import normalize_weights


def crisp_solve(
    cov: np.ndarray,
    mu: np.ndarray,
    gamma: float = 0.5,
    max_sweeps: int = 100,
    tol: float = 1e-10,
    w0: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Solve P_gamma w = mu by Gauss-Seidel.

    Returns the (unnormalised) solution vector and the number of sweeps taken.
    """
    cov = np.asarray(cov, dtype=float)
    mu = np.asarray(mu, dtype=float).ravel()
    n = cov.shape[0]
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")

    d = np.diag(cov).copy()
    # P_gamma: diagonal = sigma_ii (preserved); off-diagonal = gamma * Sigma_ij
    P = gamma * cov.copy()
    np.fill_diagonal(P, d)
    diag = np.diag(P).copy()
    if np.any(diag <= 0):
        raise ValueError("non-positive variance on the diagonal of Sigma")

    w = (mu / d) if w0 is None else np.asarray(w0, dtype=float).copy()

    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        w_prev = w.copy()
        for i in range(n):
            # use already-updated w[:i] and stale w[i+1:]
            s = P[i, :] @ w - P[i, i] * w[i]
            w[i] = (mu[i] - s) / diag[i]
        if np.max(np.abs(w - w_prev)) <= tol * (1.0 + np.max(np.abs(w))):
            break
    return w, sweeps


def crisp_weights(
    cov: np.ndarray,
    mu: np.ndarray,
    gamma: float = 0.5,
    max_sweeps: int = 100,
    tol: float = 1e-10,
    normalize: str = "gross",
) -> np.ndarray:
    """CRISP portfolio weights, post-processed for trading."""
    w, _ = crisp_solve(cov, mu, gamma=gamma, max_sweeps=max_sweeps, tol=tol)
    return normalize_weights(w, mode=normalize)
