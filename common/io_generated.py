from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


def save_generated_csv(path: str | Path, values: np.ndarray, features: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(features)
        for row in values:
            writer.writerow([f"{float(x):.10g}" if math.isfinite(float(x)) else "" for x in row])
