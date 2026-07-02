#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gc_garch.common.copulas import (  # noqa: E402
    fit_dynamic_pair_t_copula,
    simulate_dynamic_pair_t_innovations,
)
from gc_garch.common.io import ensure_run_dirs, load_experiment_config, save_json  # noqa: E402
from gc_garch.common.io_generated import save_generated_csv  # noqa: E402
from gc_garch.common.marginals import (  # noqa: E402
    fit_marginals,
    simulate_marginals_from_innovations,
)


def write_residual_correlation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "innovation_corr", "innovation_abs_corr", "rho_min", "rho_mean", "rho_std", "rho_max"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="gc_garch/models/pair_t_dynamic/config.json",
        help="Model-specific config path.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    run_dir = Path(config["output_dir"])
    dirs = ensure_run_dirs(run_dir)
    save_json(dirs["config"] / "resolved_config.json", config)

    features = list(config.get("features", ["sp500", "DGS10"]))
    scale_factor = float(config.get("scale_factor", 100.0))
    generated_length = int(config.get("generated_length", 1260))
    burn_in = int(config.get("burn_in", 750))
    num_generated = int(config.get("num_generated", 20))
    seed = int(config.get("seed", 20260702))

    marginal_fit = fit_marginals(config)
    pair_fit = fit_dynamic_pair_t_copula(marginal_fit["standardized_residuals"], config)

    rng = np.random.default_rng(seed)
    correlation_rows: list[dict[str, Any]] = []
    for idx in range(1, num_generated + 1):
        innovations, rho_path = simulate_dynamic_pair_t_innovations(
            length=generated_length + burn_in,
            pair_fit=pair_fit,
            rng=rng,
        )
        simulated_scaled = simulate_marginals_from_innovations(
            params_by_feature=marginal_fit["params_by_feature"],
            features=features,
            innovations=innovations,
            burn_in=burn_in,
        )
        simulated = simulated_scaled / scale_factor
        output_path = dirs["generated"] / f"gc_generated_{idx:03d}.csv"
        save_generated_csv(output_path, simulated, features)

        kept_innovations = innovations[burn_in:]
        kept_rhos = rho_path[burn_in:]
        corr = float(np.corrcoef(kept_innovations.T)[0, 1])
        abs_corr = float(np.corrcoef(np.abs(kept_innovations).T)[0, 1])
        correlation_rows.append(
            {
                "file": output_path.name,
                "innovation_corr": f"{corr:.10g}",
                "innovation_abs_corr": f"{abs_corr:.10g}",
                "rho_min": f"{float(np.min(kept_rhos)):.10g}",
                "rho_mean": f"{float(np.mean(kept_rhos)):.10g}",
                "rho_std": f"{float(np.std(kept_rhos, ddof=1)):.10g}",
                "rho_max": f"{float(np.max(kept_rhos)):.10g}",
            }
        )

    write_residual_correlation_csv(
        dirs["data"] / "generated_dynamic_correlations.csv",
        correlation_rows,
    )

    payload = {
        "model": "pair_t_dynamic",
        "description": "Two-series dynamic Student-t pair-copula with per-series GJR-GARCH(1,1) marginals.",
        "train_csv": config["train_csv"],
        "features": features,
        "scale_factor": scale_factor,
        "generated_length": generated_length,
        "burn_in": burn_in,
        "num_generated": num_generated,
        "seed": seed,
        "candidate_selection": {"enabled": False},
        "marginals": marginal_fit["params_json"],
        "persistence_summary": marginal_fit["persistence_summary"],
        "pair_copula": pair_fit,
        "n_train_rows": len(marginal_fit["train"]),
    }
    save_json(dirs["data"] / "fitted_params.json", payload)

    log = "\n".join(
        [
            "# pair_t_dynamic generation",
            f"run_dir: {run_dir}",
            f"train_csv: {config['train_csv']}",
            f"features: {', '.join(features)}",
            f"num_generated: {num_generated}",
            f"generated_length: {generated_length}",
            f"burn_in: {burn_in}",
            f"phi_bar: {pair_fit['phi_bar']:.6f}",
            f"a: {pair_fit['a']:.6f}",
            f"b: {pair_fit['b']:.6f}",
            f"degrees_of_freedom: {pair_fit['degrees_of_freedom']:.6f}",
            f"signal_window: {pair_fit['signal_window']}",
            "candidate_selection: disabled",
        ]
    )
    (dirs["logs"] / "generate.log").write_text(log + "\n", encoding="utf-8")
    print(log)


if __name__ == "__main__":
    main()
