# pair_t_dynamic

This model is the first dynamic two-series GC-GARCH baseline.

- Marginals: per-series GJR-GARCH(1,1)
- Dependence: dynamic bivariate Student-t pair copula
- Dynamic update: `rho_t = (1 - a - b) * phi_bar + a * xi_{t-1} + b * rho_{t-1}`
- Generation guardrails: `xi_t` can be shrunk and `rho_t` can be clipped to avoid
  finite-sample correlation paths that are too extreme for 1260-row samples
- Generation: 20 samples are generated directly; no candidate selection is used

For two variables, the graphical pair-copula construction reduces to one
dynamic copula link between `sp500` and `DGS10`.
