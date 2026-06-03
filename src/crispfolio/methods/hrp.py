"""Hierarchical Risk Parity and its signal-aware extensions.

- ``hrp_weights``        : classical HRP (Lopez de Prado, 2016) -- signal-blind.
- ``hrp_mu_weights``     : HRP-mu, allocates between sibling clusters with a 2x2
                           mean-variance step using signed inverse-variance
                           cluster representatives (Wuebben 2026, Sec. 3).
- ``hrp_sigma_mu_weights``: HRP-Sigma-mu, same skeleton but cluster
                           representatives come from a *local* mean-variance
                           solve and budgets use L1 (ray-invariant, sign-
                           preserving) normalisation (Wuebben 2026, Sec. 4).

The dendrogram construction and quasi-diagonalisation are identical across all
three -- only the per-split budgeting rule changes.  The signal-aware variants
are faithful reconstructions of the paper's description.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from ..estimators import to_corr
from .common import normalize_weights


# --------------------------------------------------------------------------- #
# Shared dendrogram machinery
# --------------------------------------------------------------------------- #
def _quasi_diag(link: np.ndarray) -> list[int]:
    """Return leaf order so that similar assets sit next to each other."""
    link = link.astype(int)
    n = link[-1, 3]  # total number of original items
    order = [link[-1, 0], link[-1, 1]]
    while max(order) >= n:  # while there are still merged clusters to expand
        new = []
        for item in order:
            if item < n:
                new.append(item)
            else:
                row = item - n
                new.append(link[row, 0])
                new.append(link[row, 1])
        order = new
    return order


def _leaf_order(corr: np.ndarray, method: str = "single") -> list[int]:
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method=method)
    return _quasi_diag(link)


def _ivp(cov_slice: np.ndarray) -> np.ndarray:
    """Inverse-variance weights within a cluster (sum to 1)."""
    iv = 1.0 / np.diag(cov_slice)
    return iv / iv.sum()


def _bisect(order: list[int]) -> list[list[int]]:
    """One level of de Prado's bisection: split every multi-item cluster."""
    out = []
    for c in order:
        if len(c) > 1:
            half = len(c) // 2
            out.append(c[:half])
            out.append(c[half:])
    return out


# --------------------------------------------------------------------------- #
# Classical HRP
# --------------------------------------------------------------------------- #
def hrp_weights(cov: np.ndarray, normalize: str = "long_only") -> np.ndarray:
    corr, _ = to_corr(cov)
    order = _leaf_order(corr)
    n = cov.shape[0]
    w = np.ones(n)

    clusters = [order]
    while clusters:
        clusters = _bisect(clusters)
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            v_l = _ivp(cov[np.ix_(left, left)]) @ cov[np.ix_(left, left)] @ _ivp(cov[np.ix_(left, left)])
            v_r = _ivp(cov[np.ix_(right, right)]) @ cov[np.ix_(right, right)] @ _ivp(cov[np.ix_(right, right)])
            alpha = 1.0 - v_l / (v_l + v_r)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
    return normalize_weights(w, mode=normalize)


# --------------------------------------------------------------------------- #
# Signal-aware variants
# --------------------------------------------------------------------------- #
def _signed_ivp_rep(cov_s: np.ndarray, mu_s: np.ndarray) -> np.ndarray:
    """HRP-mu cluster representative: a_i ~ sign(mu_i) / sigma_ii, L1-normalised."""
    a = np.sign(mu_s) * (1.0 / np.diag(cov_s))
    s = np.abs(a).sum()
    return a / s if s > 0 else a


def _local_mv_rep(cov_s: np.ndarray, mu_s: np.ndarray) -> np.ndarray:
    """HRP-Sigma-mu cluster representative: local MV solve, L1-normalised."""
    a = np.linalg.pinv(cov_s) @ mu_s
    s = np.abs(a).sum()
    return a / s if s > 0 else a


def _hrp_signal(cov, mu, rep_fn, normalize) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    mu = np.asarray(mu, dtype=float).ravel()
    corr, _ = to_corr(cov)
    order = _leaf_order(corr)
    n = cov.shape[0]
    w = np.ones(n)

    clusters = [order]
    while clusters:
        clusters = _bisect(clusters)
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            a_l = rep_fn(cov[np.ix_(left, left)], mu[left])
            a_r = rep_fn(cov[np.ix_(right, right)], mu[right])

            # Reduce the two clusters to a 2x2 mean-variance problem.
            m = np.array([a_l @ mu[left], a_r @ mu[right]])
            s_ll = a_l @ cov[np.ix_(left, left)] @ a_l
            s_rr = a_r @ cov[np.ix_(right, right)] @ a_r
            s_lr = a_l @ cov[np.ix_(left, right)] @ a_r
            sigma2 = np.array([[s_ll, s_lr], [s_lr, s_rr]])

            beta = np.linalg.pinv(sigma2) @ m  # 2x2 MV direction
            # Between-cluster budgets are non-negative magnitudes, L1-normalised
            # (ray-invariant); the sign lives on the leaves (see below).
            budget = np.abs(beta)
            denom = budget.sum()
            budget = np.array([0.5, 0.5]) if denom <= 1e-15 else budget / denom
            w[left] *= budget[0]
            w[right] *= budget[1]

    # Each final weight factors as sign(mu_i) x (product of non-negative budgets).
    w *= np.sign(mu)
    return normalize_weights(w, mode=normalize)


def hrp_mu_weights(cov, mu, normalize: str = "gross") -> np.ndarray:
    return _hrp_signal(cov, mu, _signed_ivp_rep, normalize)


def hrp_sigma_mu_weights(cov, mu, normalize: str = "gross") -> np.ndarray:
    return _hrp_signal(cov, mu, _local_mv_rep, normalize)
