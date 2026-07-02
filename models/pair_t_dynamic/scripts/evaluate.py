#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gc_garch.common.io import ensure_run_dirs, load_experiment_config  # noqa: E402


def copy_generated_as_masks(generated_dir: Path, mask_dir: Path) -> list[Path]:
    mask_dir.mkdir(parents=True, exist_ok=True)
    for stale in mask_dir.glob("mask*_gc.csv"):
        stale.unlink()

    written: list[Path] = []
    for idx, path in enumerate(sorted(generated_dir.glob("*.csv")), start=1):
        output_path = mask_dir / f"mask{idx}_gc.csv"
        shutil.copy2(path, output_path)
        written.append(output_path)
    if not written:
        raise FileNotFoundError(f"No generated CSV files found in {generated_dir}")
    return written


def relabel_generator_column(path: Path, generator: str) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "generator" not in reader.fieldnames:
            return
        rows = list(reader)
        fields = reader.fieldnames
    for row in rows:
        row["generator"] = generator
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relabel_outputs(output_dir: Path, generator: str) -> None:
    for path in [
        output_dir / "features" / "each_mask_features.csv",
        output_dir / "results" / "feature_zscores.csv",
        output_dir / "results" / "mahalanobis_distances.csv",
        output_dir / "results" / "mask_distance_positions.csv",
    ]:
        relabel_generator_column(path, generator)


def patch_summary(output_dir: Path, title: str, generator: str) -> None:
    summary_path = output_dir / "results" / "summary.txt"
    if not summary_path.exists():
        return
    summary = summary_path.read_text(encoding="utf-8")
    summary = summary.replace("# TimeGAN Mahalanobis evaluation result", title, 1)
    summary = summary.replace("# Mahalanobis remake2 result", title, 1)
    summary = summary.replace(",unknown,", f",{generator},")
    summary = summary.replace(",garch,", f",{generator},")
    summary_path.write_text(summary, encoding="utf-8")


def run_evaluator(
    script: Path,
    train_csv: str,
    mask_dir: Path,
    output_dir: Path,
    window_length: int,
    stride: int,
) -> None:
    cmd = [
        sys.executable,
        str(script),
        "--train-csv",
        train_csv,
        "--mask-dir",
        str(mask_dir),
        "--output-dir",
        str(output_dir),
        "--window-length",
        str(window_length),
        "--stride",
        str(stride),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


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
    generated_dir = dirs["generated"]
    mask_dir = dirs["evaluation"] / "mahalanobis_input"
    copied = copy_generated_as_masks(generated_dir, mask_dir)
    print(f"Prepared {len(copied)} generated files for Mahalanobis evaluation")

    window_length = int(config.get("window_length", 1260))
    stride = int(config.get("stride", 126))
    train_csv = str(config["train_csv"])
    repo_root = Path(config["repo_root"])

    origin_output = dirs["evaluation"] / "mahalanobis_results"
    remake2_output = dirs["evaluation"] / "mahalanobis_remake2_results"
    run_evaluator(
        script=repo_root / str(config["origin_script"]),
        train_csv=train_csv,
        mask_dir=mask_dir,
        output_dir=origin_output,
        window_length=window_length,
        stride=stride,
    )
    relabel_outputs(origin_output, "gc_garch_dynamic")
    patch_summary(origin_output, "# GC-GARCH dynamic Mahalanobis evaluation result", "gc_garch_dynamic")

    run_evaluator(
        script=repo_root / str(config["remake2_script"]),
        train_csv=train_csv,
        mask_dir=mask_dir,
        output_dir=remake2_output,
        window_length=window_length,
        stride=stride,
    )
    relabel_outputs(remake2_output, "gc_garch_dynamic")
    patch_summary(remake2_output, "# GC-GARCH dynamic Mahalanobis remake2 result", "gc_garch_dynamic")

    log = "\n".join(
        [
            "# pair_t_dynamic evaluation",
            f"run_dir: {run_dir}",
            f"mask_dir: {mask_dir}",
            f"origin_output: {origin_output}",
            f"remake2_output: {remake2_output}",
            f"window_length: {window_length}",
            f"stride: {stride}",
        ]
    )
    (dirs["logs"] / "evaluate.log").write_text(log + "\n", encoding="utf-8")
    print(log)


if __name__ == "__main__":
    main()
