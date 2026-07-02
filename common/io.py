from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def gc_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(model_config_path: str | Path) -> dict[str, Any]:
    root = repo_root()
    gc = gc_root()
    config = {}
    for rel in [
        "config/shared_defaults.json",
        "config/data.json",
        "config/mahalanobis.json",
    ]:
        config = deep_merge(config, load_json(gc / rel))
    config = deep_merge(config, load_json(model_config_path))
    config["repo_root"] = str(root)
    config["gc_root"] = str(gc)
    return config


def ensure_run_dirs(run_dir: str | Path) -> dict[str, Path]:
    root = Path(run_dir)
    dirs = {
        "root": root,
        "config": root / "config",
        "data": root / "data",
        "generated": root / "generated",
        "evaluation": root / "evaluation",
        "logs": root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
