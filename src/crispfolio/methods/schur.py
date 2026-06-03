"""Cotton (2024) Schur Complementary Allocation. arXiv:2411.05807.

Recursive bisection on the HRP dendrogram, but at each split the two child
covariance blocks are *augmented* with off-diagonal information through a
gamma-scaled Schur complement.  This interpolates between HRP (gamma = 0) and
the minimum-variance portfolio (gamma = 1).  The method is signal-blind: it
only solves the minimum-variance problem.

At a node, partition Sigma = [[A, B], [C=B', D]] (A = left cluster, D = right):

    A^c = A - gamma B D^{-1} C          b_A = 1 - gamma B D^{-1} 1
    nu(A') = 1 / (b_A' (A^c)^{-1} b_A)   (augmented cluster variance)
    A''   = A^c / (b_A b_A')             (element-wise; passed down for recursion)

and symmetrically for D.  Capital is split inversely to the augmented cluster
variances and combined with the recursively-computed intra-cluster weights.
"""
from __future__ import annotations

import numpy as np

from ..estimators import to_corr
from .common import normalize_weights
from .hrp import _leaf_order

_EPS = 1e-12


def _pinv(m: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(np.atleast_2d(m))


def _aug_variance(Ac: np.ndarray, b: np.ndarray) -> float:
    q = float(b @ _pinv(Ac) @ b)
    # Guard against numerical non-positivity (can happen near gamma = 1).
    return 1.0 / q if q > _EPS else np.inf


def _recurse(cov: np.ndarray, gamma: float) -> np.ndarray:
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    half = n // 2
    A = cov[:half, :half]
    D = cov[half:, half:]
    B = cov[:half, half:]
    C = B.T
    one_A = np.ones(half)
    one_D = np.ones(n - half)

    Ainv = _pinv(A)
    Dinv = _pinv(D)

    Ac = A - gamma * (B @ Dinv @ C)
    Dc = D - gamma * (C @ Ainv @ B)
    bA = one_A - gamma * (B @ Dinv @ one_D)
    bD = one_D - gamma * (C @ Ainv @ one_A)

    nu_A = _aug_variance(Ac, bA)
    nu_D = _aug_variance(Dc, bD)

    # Element-wise augmented blocks passed to the children.  Protect against
    # vanishing b entries when gamma -> 1.
    outer_A = np.outer(bA, bA)
    outer_D = np.outer(bD, bD)
    outer_A = np.where(np.abs(outer_A) < _EPS, _EPS, outer_A)
    outer_D = np.where(np.abs(outer_D) < _EPS, _EPS, outer_D)
    A_pp = Ac / outer_A
    D_pp = Dc / outer_D

    w_A = _recurse(A_pp, gamma)
    w_D = _recurse(D_pp, gamma)

    inv_A = 0.0 if not np.isfinite(nu_A) else 1.0 / nu_A
    inv_D = 0.0 if not np.isfinite(nu_D) else 1.0 / nu_D
    total = inv_A + inv_D
    if total <= _EPS:
        alpha_A = alpha_D = 0.5
    else:
        alpha_A, alpha_D = inv_A / total, inv_D / total

    return np.concatenate([alpha_A * w_A, alpha_D * w_D])


def schur_weights(cov: np.ndarray, gamma: float = 0.5, normalize: str = "long_only") -> np.ndarray:
    """Cotton Schur-complement minimum-variance weights."""
    cov = np.asarray(cov, dtype=float)
    corr, _ = to_corr(cov)
    order = _leaf_order(corr)
    ordered = cov[np.ix_(order, order)]

    w_ordered = _recurse(ordered, gamma)

    # unscramble back to the original asset order
    w = np.empty(cov.shape[0])
    w[order] = w_ordered
    return normalize_weights(w, mode=normalize)
