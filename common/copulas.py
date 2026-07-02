from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

from .io import repo_root

GARCH_SCRIPT_DIR = repo_root() / "garch_origin" / "scripts"
if str(GARCH_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(GARCH_SCRIPT_DIR))

from garch_utils import estimate_t_copula_parameters, t_copula_innovations  # noqa: E402


def _clip_rho(value: float, limit: float = 0.98) -> float:
    return float(np.clip(value, -limit, limit))


def df_grid_from_config(config: dict[str, Any]) -> list[float] | None:
    copula_config = dict(config.get("pair_copula", {}))
    if "df_grid" in copula_config:
        return [float(value) for value in copula_config["df_grid"]]
    if all(key in copula_config for key in ["df_min", "df_max", "df_step"]):
        df_min = float(copula_config["df_min"])
        df_max = float(copula_config["df_max"])
        df_step = float(copula_config["df_step"])
        count = int(round((df_max - df_min) / df_step))
        return [
            round(df_min + idx * df_step, 10)
            for idx in range(count + 1)
            if df_min + idx * df_step <= df_max + 1e-9
        ]
    return None


def fit_static_pair_t_copula(residual_matrix: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    copula_config = dict(config.get("pair_copula", {}))
    fit = estimate_t_copula_parameters(
        residual_matrix,
        df_values=df_grid_from_config(config),
        rho_min=float(copula_config.get("rho_min", -0.5)),
        rho_max=float(copula_config.get("rho_max", 0.5)),
        rho_step=float(copula_config.get("rho_step", 0.01)),
    )
    return {
        "type": "static_t_pair_copula",
        "rho": float(fit["rho"]),
        "degrees_of_freedom": float(fit["degrees_of_freedom"]),
        "correlation_matrix": fit["correlation_matrix"],
        "loglik": float(fit["loglik"]),
        "n_observations": int(fit["n_observations"]),
        "fit_summary": fit,
    }


def simulate_static_pair_t_innovations(
    length: int,
    pair_fit: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    return t_copula_innovations(
        length=length,
        corr=np.asarray(pair_fit["correlation_matrix"], dtype=np.float64),
        degrees_of_freedom=float(pair_fit["degrees_of_freedom"]),
        rng=rng,
        exact_latent_sample_corr=False,
    )


def local_correlation_signal(values: np.ndarray, end: int, window: int) -> float:
    start = max(0, end - window)
    sample = values[start:end]
    if sample.shape[0] < 2:
        return 0.0
    sample = sample - sample.mean(axis=0, keepdims=True)
    denom = float(np.sqrt(np.sum(sample[:, 0] ** 2) * np.sum(sample[:, 1] ** 2)))
    if denom <= 1e-12:
        return 0.0
    return _clip_rho(float(np.sum(sample[:, 0] * sample[:, 1]) / denom))


def dynamic_rho_path(
    values: np.ndarray,
    phi_bar: float,
    a: float,
    b: float,
    signal_window: int,
    rho_limit: float = 0.98,
    signal_shrink: float = 1.0,
) -> np.ndarray:
    n = int(values.shape[0])
    rhos = np.empty(n, dtype=np.float64)
    rhos[0] = _clip_rho(phi_bar, rho_limit)
    for t in range(1, n):
        xi = signal_shrink * local_correlation_signal(values, end=t, window=signal_window)
        rhos[t] = _clip_rho((1.0 - a - b) * phi_bar + a * xi + b * rhos[t - 1], rho_limit)
    return rhos


def dynamic_scaled_bivariate_t_loglik(
    values: np.ndarray,
    rhos: np.ndarray,
    degrees_of_freedom: float,
) -> float:
    if values.shape[1] != 2:
        raise ValueError("dynamic_scaled_bivariate_t_loglik expects exactly two columns")
    if degrees_of_freedom <= 2.0:
        return -math.inf

    nu = float(degrees_of_freedom)
    scale = math.sqrt((nu - 2.0) / nu)
    x = values / scale
    rho = np.clip(rhos, -0.98, 0.98)
    det = np.maximum(1.0 - rho * rho, 1e-12)
    q = (x[:, 0] ** 2 - 2.0 * rho * x[:, 0] * x[:, 1] + x[:, 1] ** 2) / det
    constant = (
        math.lgamma((nu + 2.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - math.log(nu * math.pi)
        - np.log(scale) * 2.0
        - 0.5 * np.log(det)
    )
    return float(np.sum(constant - ((nu + 2.0) / 2.0) * np.log1p(q / nu)))


def fit_dynamic_pair_t_copula(residual_matrix: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(residual_matrix, dtype=np.float64)
    values = values[np.all(np.isfinite(values), axis=1)]
    values = values - values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, ddof=1)
    std[std <= 0.0] = 1.0
    values = values / std

    static_fit = fit_static_pair_t_copula(values, config)
    dynamic_config = dict(config.get("dynamic_pair_copula", {}))
    signal_window = int(dynamic_config.get("signal_window", 20))
    rho_limit = float(dynamic_config.get("rho_limit", 0.98))
    signal_shrink = float(dynamic_config.get("signal_shrink", 1.0))
    phi_bar = _clip_rho(float(static_fit["rho"]))
    nu = float(static_fit["degrees_of_freedom"])
    a_grid = [float(value) for value in dynamic_config.get("a_grid", [0.0, 0.02, 0.05, 0.10])]
    b_grid = [float(value) for value in dynamic_config.get("b_grid", [0.0, 0.50, 0.80, 0.92, 0.97])]

    best: dict[str, Any] = {
        "a": 0.0,
        "b": 0.0,
        "loglik": dynamic_scaled_bivariate_t_loglik(
            values,
            np.full(values.shape[0], phi_bar, dtype=np.float64),
            nu,
        ),
    }
    for a in a_grid:
        for b in b_grid:
            if a < 0.0 or b < 0.0 or a + b >= 0.999:
                continue
            rhos = dynamic_rho_path(
                values,
                phi_bar=phi_bar,
                a=a,
                b=b,
                signal_window=signal_window,
                rho_limit=rho_limit,
                signal_shrink=signal_shrink,
            )
            loglik = dynamic_scaled_bivariate_t_loglik(values, rhos, nu)
            if loglik > float(best["loglik"]):
                best = {"a": a, "b": b, "loglik": loglik}

    fitted_path = dynamic_rho_path(
        values,
        phi_bar=phi_bar,
        a=float(best["a"]),
        b=float(best["b"]),
        signal_window=signal_window,
        rho_limit=rho_limit,
        signal_shrink=signal_shrink,
    )
    return {
        "type": "dynamic_t_pair_copula",
        "phi_bar": phi_bar,
        "a": float(best["a"]),
        "b": float(best["b"]),
        "degrees_of_freedom": nu,
        "signal_window": signal_window,
        "rho_limit": rho_limit,
        "signal_shrink": signal_shrink,
        "static_fit": static_fit,
        "loglik": float(best["loglik"]),
        "rho_path_summary": {
            "min": float(np.min(fitted_path)),
            "mean": float(np.mean(fitted_path)),
            "std": float(np.std(fitted_path, ddof=1)),
            "max": float(np.max(fitted_path)),
        },
    }


def simulate_dynamic_pair_t_innovations(
    length: int,
    pair_fit: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if length < 2:
        raise ValueError("length must be at least 2")
    nu = float(pair_fit["degrees_of_freedom"])
    if nu <= 2.0:
        raise ValueError("degrees_of_freedom must be greater than 2.0")
    phi_bar = _clip_rho(float(pair_fit["phi_bar"]))
    a = float(pair_fit["a"])
    b = float(pair_fit["b"])
    signal_window = int(pair_fit["signal_window"])
    rho_limit = float(pair_fit.get("rho_limit", 0.98))
    signal_shrink = float(pair_fit.get("signal_shrink", 1.0))
    scale = math.sqrt((nu - 2.0) / nu)

    innovations = np.zeros((length, 2), dtype=np.float64)
    rhos = np.empty(length, dtype=np.float64)
    rhos[0] = phi_bar
    for t in range(length):
        if t > 0:
            xi = signal_shrink * local_correlation_signal(innovations, end=t, window=signal_window)
            rhos[t] = _clip_rho((1.0 - a - b) * phi_bar + a * xi + b * rhos[t - 1], rho_limit)
        rho = rhos[t]
        z0 = rng.standard_normal()
        z1 = rho * z0 + math.sqrt(max(1.0 - rho * rho, 1e-12)) * rng.standard_normal()
        common_scale = math.sqrt(rng.chisquare(nu) / nu)
        innovations[t, 0] = (z0 / max(common_scale, 1e-12)) * scale
        innovations[t, 1] = (z1 / max(common_scale, 1e-12)) * scale
    return innovations, rhos
