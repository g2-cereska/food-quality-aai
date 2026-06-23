"""Aggregates outputs/<architecture>/metrics.json across all trained
models into a single comparison table.

Usage:
    python -m src.compare_models

Writes outputs/model_comparison.csv and prints a summary to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

COLUMNS = [
    "architecture",
    "test_accuracy",
    "test_macro_f1",
    "parameters.total_params",
    "training_time_seconds",
    "avg_inference_ms_per_image",
]


def load_all_metrics() -> list[dict]:
    """Loads every outputs/<architecture>/metrics.json found on disk."""
    records = []
    for metrics_path in sorted(OUTPUTS_DIR.glob("*/metrics.json")):
        with open(metrics_path) as f:
            records.append(json.load(f))
    if not records:
        raise FileNotFoundError(
            "No metrics.json files found under outputs/. "
            "Run src.train for at least one architecture first."
        )
    return records


def build_comparison_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append({
            "architecture": record["architecture"],
            "test_accuracy": record["test_accuracy"],
            "test_macro_f1": record["test_macro_f1"],
            "total_params": record["parameters"]["total_params"],
            "training_time_seconds": record["training_time_seconds"],
            "avg_inference_ms_per_image": record["avg_inference_ms_per_image"],
        })
    df = pd.DataFrame(rows).set_index("architecture")
    return df.sort_values("test_macro_f1", ascending=False)


def main() -> None:
    records = load_all_metrics()
    table = build_comparison_table(records)

    print("\nModel comparison (sorted by macro F1):\n")
    print(table.to_string())

    output_path = OUTPUTS_DIR / "model_comparison.csv"
    table.to_csv(output_path)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()