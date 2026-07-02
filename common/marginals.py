from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .io import repo_root

GARCH_SCRIPT_DIR = repo_root() / "garch_origin" / "scripts"
if str(GARCH_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(GARCH_SCRIPT_DIR))

from garch_utils import (  # noqa: E402
    fit_garch11,
    fit_gjr_garch11,
    fit_spec_from_config,
    params_to_json,
    read_train_csv,
)


def fit_marginals(config: dict[str, Any]) -> dict[str, Any]:
    features = list(config.get("features", ["sp500", "DGS10"]))
    scale_factor = float(config.get("scale_factor", 100.0))
    train = read_train_csv(config["train_csv"], features)
    scaled = train.to_numpy(dtype=np.float64) * scale_factor
    volatility_model = str(config.get("volatility_model", "gjr_garch11")).lower()

    params_by_feature = {}
    residuals = []
    persistence = {}
    for idx, feature in enumerate(features):
        spec = fit_spec_from_config(config, feature)
        if volatility_model in {"gjr", "gjr_garch", "gjr_garch11", "gjr-garch11"}:
            gjr_config = dict(config.get("gjr_garch", {}))
            params, variances, std_residuals = fit_gjr_garch11(
                scaled[:, idx],
                spec,
                gamma_values=[
                    float(value)
                    for value in gjr_config.get(
                        "gamma_grid",
                        [0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.26],
                    )
                ],
            )
            effective_persistence = params.alpha + 0.5 * params.gamma + params.beta
        elif volatility_model in {"garch", "garch11", "garch-garch11"}:
            params, variances, std_residuals = fit_garch11(scaled[:, idx], spec)
            effective_persistence = params.alpha + params.beta
        else:
            raise ValueError(f"Unsupported volatility_model: {volatility_model}")
        params_by_feature[feature] = params
        residuals.append(std_residuals)
        persistence[feature] = {
            "alpha_plus_beta": float(params.alpha + params.beta),
            "alpha_plus_half_gamma_plus_beta": float(effective_persistence),
            "gjr_gamma": float(params.gamma),
            "volatility_half_life_lags": None
            if effective_persistence <= 0.0 or effective_persistence >= 1.0
            else float(math.log(0.5) / math.log(effective_persistence)),
        }

    residual_matrix = np.column_stack(residuals)
    return {
        "train": train,
        "features": features,
        "scale_factor": scale_factor,
        "params_by_feature": params_by_feature,
        "params_json": params_to_json(params_by_feature),
        "standardized_residuals": residual_matrix,
        "persistence_summary": persistence,
    }


def simulate_marginals_from_innovations(
    params_by_feature: dict[str, Any],
    features: list[str],
    innovations: np.ndarray,
    burn_in: int,
) -> np.ndarray:
    total_length = int(innovations.shape[0])
    values = np.zeros((total_length, len(features)), dtype=np.float64)
    residuals = np.zeros_like(values)
    variances = np.zeros_like(values)

    for col, feature in enumerate(features):
        params = params_by_feature[feature]
        variances[0, col] = max(float(params.unconditional_variance), 1e-10)
        values[0, col] = float(params.mu)

    for t in range(1, total_length):
        for col, feature in enumerate(features):
            params = params_by_feature[feature]
            if params.volatility_model == "gjr_garch11":
                leverage = (
                    params.gamma * residuals[t - 1, col] ** 2
                    if residuals[t - 1, col] < 0.0
                    else 0.0
                )
                variances[t, col] = (
                    params.omega
                    + params.alpha * residuals[t - 1, col] ** 2
                    + leverage
                    + params.beta * variances[t - 1, col]
                )
            else:
                variances[t, col] = (
                    params.omega
                    + params.alpha * residuals[t - 1, col] ** 2
                    + params.beta * variances[t - 1, col]
                )
            variances[t, col] = max(variances[t, col], 1e-10)
            residuals[t, col] = math.sqrt(variances[t, col]) * innovations[t, col]
            values[t, col] = params.mu + residuals[t, col]

    return values[burn_in:]
