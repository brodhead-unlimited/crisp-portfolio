"""crispfolio — general mean-variance portfolio construction.

Implements the methods from Wuebben (2026), "Beyond De Prado and Cotton:
Hierarchical and Iterative Methods for General Mean-Variance Portfolios"
(arXiv:2604.23833), plus the standard baselines they are compared against.

Methods
-------
- crisp           : Correlation-Regularised Iterative Shrinkage Portfolios
- hrp             : standard Hierarchical Risk Parity (de Prado 2016)
- hrp_mu          : signal-aware HRP
- hrp_sigma_mu    : signal-aware HRP with recursive local mean-variance
- schur           : Cotton (2024) Schur-complement allocation
- baselines       : 1/N, Markowitz, minimum-variance
"""

__version__ = "0.1.0"
