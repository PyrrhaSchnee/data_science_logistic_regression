#!/usr/bin/env python3
"""
describe.py

Takes a dataset as an argument then displays
information for all numerical features.
"""

from __future__ import annotations

import sys
from pathlib import Path

from utilities import (
    Dataset,
    column_has_number,
    fill_column_with_number,
    get_25_quartile,
    get_50_quartile,
    get_75_quartile,
    get_mean,
    get_std,
    get_csv_path,
    load_dataset,
)


def compute_stats(values: list[float]) -> dict[str, float]:
    """
    Compute the required stats using the project's intended rules.
    """
    if not values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q1": 0.0,
            "q2": 0.0,
            "q3": 0.0,
            "max": 0.0,
        }

    sorted_v = sorted(values)
    return {
        "count": float(len(sorted_v)),
        "mean": get_mean(sorted_v),
        "std": get_std(sorted_v),
        "min": float(sorted_v[0]),
        "q1": get_25_quartile(sorted_v),
        "q2": get_50_quartile(sorted_v),
        "q3": get_75_quartile(sorted_v),
        "max": float(sorted_v[-1]),
    }


def compute_widths(
    columns: list[str], stats_by_column: list[dict[str, float]]
) -> list[int]:
    """
    Compute a width per column so header + all printed values fit.
    """
    keys = ["count", "mean", "std", "min", "q1", "q2", "q3", "max"]
    widths: list[int] = []

    for col, st in zip(columns, stats_by_column):
        w = len(col)
        for k in keys:
            s = f"{st[k]:.6f}"
            if len(s) > w:
                w = len(s)
        widths.append(w)

    return widths


def print_header(columns: list[str], widths: list[int], label_w: int) -> None:
    """
    Print the header row with per-column widths.
    """
    parts: list[str] = ["".ljust(label_w)]
    for name, w in zip(columns, widths):
        parts.append(" " + name.rjust(w))
    print("".join(parts))


def print_row(
    label: str, values: list[float], widths: list[int], label_w: int
) -> None:
    """
    Print one row with per-column widths.
    """
    parts: list[str] = [label.ljust(label_w)]
    for v, w in zip(values, widths):
        parts.append(" " + f"{v:.6f}".rjust(w))
    print("".join(parts))


def main() -> int:
    """
    main function
    """
    try:
        dataset_path: str | Path = get_csv_path()
        dataset: Dataset = load_dataset(dataset_path)
        assert dataset.rows, "empty dataset"
        exclude: list[str] = []
        if dataset.index_name:
            exclude.append(dataset.index_name)
        if dataset.label_name:
            exclude.append(dataset.label_name)
        columns = column_has_number(dataset, exclude=exclude)
        assert columns, "Invalid input: no numeric values in .csv file"
        stats_by_column: list[dict[str, float]] = []
        for col in columns:
            values = fill_column_with_number(dataset, col)
            stats_by_column.append(compute_stats(values))
        widths = compute_widths(columns, stats_by_column)
        row_labels = [
            "Count",
            "Mean",
            "Std",
            "Min",
            "25%",
            "50%",
            "75%",
            "Max",
        ]
        label_w = 0
        for x in row_labels:
            if len(x) > label_w:
                label_w = len(x)
        print_header(columns, widths, label_w)
        print_row(
            "Count", [s["count"] for s in stats_by_column], widths, label_w
        )
        print_row(
            "Mean", [s["mean"] for s in stats_by_column], widths, label_w
        )
        print_row("Std", [s["std"] for s in stats_by_column], widths, label_w)
        print_row("Min", [s["min"] for s in stats_by_column], widths, label_w)
        print_row("25%", [s["q1"] for s in stats_by_column], widths, label_w)
        print_row("50%", [s["q2"] for s in stats_by_column], widths, label_w)
        print_row("75%", [s["q3"] for s in stats_by_column], widths, label_w)
        print_row("Max", [s["max"] for s in stats_by_column], widths, label_w)

        return 0

    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
