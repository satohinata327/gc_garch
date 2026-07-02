# pair_t_static

This model is a two-series minimal GC-GARCH baseline.

- Marginals: per-series GJR-GARCH(1,1)
- Dependence: static bivariate Student-t pair copula
- Generation: 20 samples are generated directly; no candidate selection is used

For two variables, the pair-copula construction reduces to one bivariate copula
linking `sp500` and `DGS10`.
