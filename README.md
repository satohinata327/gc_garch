# gc_garch

Experimental Graphical Copula GARCH workspace.

This directory separates reusable infrastructure from model-specific code:

- `config/`: shared data and evaluation defaults
- `common/`: reusable helpers
- `models/`: model-specific implementations and runs

The first models are two-series pair t-copula GARCH baselines for `sp500` and
`DGS10`.

## Models

`models/pair_t_static` is the initial static baseline:

- Marginals: per-series GJR-GARCH(1,1)
- Dependence: static bivariate Student-t pair copula
- Generation: 20 samples directly, with no candidate selection

`models/pair_t_dynamic` adds a time-varying pair-copula correlation:

- Marginals: per-series GJR-GARCH(1,1)
- Dependence: dynamic bivariate Student-t pair copula
- Update: `rho_t = (1 - a - b) * phi_bar + a * xi_{t-1} + b * rho_{t-1}`
- Generation: 20 samples directly, with no candidate selection

Run it from the repository root:

```bash
.venv/bin/python gc_garch/models/pair_t_static/scripts/generate.py --config gc_garch/models/pair_t_static/config.json
.venv/bin/python gc_garch/models/pair_t_static/scripts/evaluate.py --config gc_garch/models/pair_t_static/config.json
.venv/bin/python gc_garch/models/pair_t_dynamic/scripts/generate.py --config gc_garch/models/pair_t_dynamic/config.json
.venv/bin/python gc_garch/models/pair_t_dynamic/scripts/evaluate.py --config gc_garch/models/pair_t_dynamic/config.json
```

Default outputs are written below each model's `runs/` directory:

```text
gc_garch/models/pair_t_static/runs/run_001/
gc_garch/models/pair_t_dynamic/runs/run_001/
```
