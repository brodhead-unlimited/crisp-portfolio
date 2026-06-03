"""Correctness checks for the allocators."""
import numpy as np
import pandas as pd
import pytest

from crispfolio import methods as m
from crispfolio.methods.crisp import crisp_solve


def _spd(n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    cov = a @ a.T / n + np.eye(n) * 0.5
    return cov


def test_crisp_solves_the_regularised_system():
    cov = _spd(8, seed=1)
    mu = np.random.default_rng(2).standard_normal(8)
    gamma = 0.6
    w, sweeps = crisp_solve(cov, mu, gamma=gamma, max_sweeps=500, tol=1e-12)

    P = gamma * cov.copy()
    np.fill_diagonal(P, np.diag(cov))
    w_direct = np.linalg.solve(P, mu)
    assert np.allclose(w, w_direct, atol=1e-8)
    assert sweeps < 500


def test_crisp_gamma1_matches_markowitz_direction():
    cov = _spd(6, seed=3)
    mu = np.random.default_rng(4).standard_normal(6)
    w, _ = crisp_solve(cov, mu, gamma=1.0, max_sweeps=1000, tol=1e-12)
    assert np.allclose(w, np.linalg.solve(cov, mu), atol=1e-6)


def test_crisp_gamma0_is_diagonal_rule():
    cov = _spd(6, seed=5)
    mu = np.random.default_rng(6).standard_normal(6)
    w, _ = crisp_solve(cov, mu, gamma=0.0, max_sweeps=10)
    assert np.allclose(w, mu / np.diag(cov))


def test_hrp_weights_long_only_sum_to_one():
    cov = _spd(10, seed=7)
    w = m.hrp_weights(cov)
    assert np.all(w >= -1e-12)
    assert abs(w.sum() - 1.0) < 1e-9


def test_schur_gamma0_close_to_hrp_family():
    cov = _spd(12, seed=8)
    w = m.schur_weights(cov, gamma=0.0)
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.all(w >= -1e-9)


def test_gross_normalisation():
    cov = _spd(5, seed=9)
    mu = np.random.default_rng(10).standard_normal(5)
    w = m.crisp_weights(cov, mu, gamma=0.5, normalize="gross")
    assert abs(np.abs(w).sum() - 1.0) < 1e-9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
