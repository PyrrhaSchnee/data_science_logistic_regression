#!/usr/bin/env python3
"""
scatter_plot.py

Display a scatter plot answering the following question:
What are the two features that are similiar ?
My interpretation: two features are considered "similiar"
if they have (almost) the same information.
Visually on a scatter plot they show up as points lying close to a line
We can use Pearson correlation coefficient to quantify the "similarity"
There are 13 numeric features, so 78 pairs to test

Also save the plotted image of all possible pairs into
a single file
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from utilities import (
    Dataset,
    column_has_number,
    get_csv_path,
    get_float,
    load_dataset,
    get_house_names,
)


def extract_pairs(
    ds: Dataset, x_name: str, y_name: str
) -> tuple[list[float], list[float]]:
    """
    Extract paired values (x, y) from 2 columns.
    Rows with missing/invalid value are ignored.
    """
    try:
        x_idx = ds.header.index(x_name)
        y_idx = ds.header.index(y_name)
    except Exception:
        return ([], [])

    xs: list[float] = []
    ys: list[float] = []
    for row in ds.rows:
        x = get_float(row[x_idx])
        y = get_float(row[y_idx])
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return (xs, ys)


def extract_pairs_by_house(
    ds: Dataset, houses: list[str], x_name: str, y_name: str
) -> dict[str, tuple[list[float], list[float]]]:
    """
    Extract paired values (x, y) grouped by house.
    Rows with missing/invalid value are ignored.
    If house is empty return am empty dict.
    """
    if not houses:
        return {}
    if ds.label_name is None:
        return {}
    try:
        x_idx = ds.header.index(x_name)
        y_idx = ds.header.index(y_name)
        # h = house
        h_idx = ds.header.index(ds.label_name)
    except Exception:
        return {}
    ret: dict[str, tuple[list[float], list[float]]] = {}
    for h in houses:
        ret[h] = ([], [])
    for row in ds.rows:
        house = row[h_idx].strip()
        if house not in ret:
            continue
        x = get_float(row[x_idx])
        y = get_float(row[y_idx])
        if x is None or y is None:
            continue
        ret[house][0].append(x)
        ret[house][1].append(y)
    return ret


def get_pcc(xs: list[float], ys: list[float]) -> float | None:
    """
    Compute Pearson correlation coefficient for 2 lists of floats.
    Returns None is not enough data or if variance is 0.
    """
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = 0.0
    mean_y = 0.0
    i = 0
    while i < n:
        mean_x += xs[i]
        mean_y += ys[i]
        i += 1
    mean_x /= float(n)
    mean_y /= float(n)
    num = 0.0
    # sx means sum of squared deviations of x from its mean
    sx = 0.0
    sy = 0.0
    i = 0
    while i < n:
        dx = xs[i] - mean_x
        dy = ys[i] - mean_y
        num += dx * dy
        sx += dx * dx
        sy += dy * dy
        i += 1
    # de = denominator
    de = math.sqrt(sx) * math.sqrt(sy)
    if de == 0.0:
        return None
    return num / de


def choose_most_similiar_pair(
    ds: Dataset, features: list[str]
) -> tuple[str, str, float, int] | None:
    """
    Find the pair of features with the highest absolute
    Pearson correlation coefficient r.
    Return (x_name, y_name, r, n_points) or None.
    """
    best_x = None
    best_y = None
    best_r = None
    best_abs = None
    best_n = 0
    i = 0
    while i < len(features):
        j = i + 1
        while j < len(features):
            a = features[i]
            b = features[j]
            xs, ys = extract_pairs(ds, a, b)
            r = get_pcc(xs, ys)
            if r is not None:
                abs_r = abs(r)
                if best_abs is None or abs_r > best_abs:
                    best_abs = abs_r
                    best_r = r
                    best_x = a
                    best_y = b
                    best_n = len(xs)
            j += 1
        i += 1
    if best_x is None or best_y is None or best_r is None:
        return None
    return (best_x, best_y, best_r, best_n)


def scatter_plot(
    ds: Dataset, houses: list[str], x_name: str, y_name: str, r: float
) -> None:
    """
    Scatter plot for (x_name, y_name)
    If houses exist, colour points by house
    """
    plt.figure()
    if houses:
        data = extract_pairs_by_house(ds, houses, x_name, y_name)
        for h in houses:
            xs, ys = data.get(h, ([], []))
            plt.scatter(xs, ys, label=h, alpha=0.6, s=12)
        plt.legend()
    else:
        xs, ys = extract_pairs(ds, x_name, y_name)
        plt.scatter(xs, ys, alpha=0.6, s=12)
    plt.title(f"Most similiar features: {x_name} vs {y_name} (r={r:.6f})")
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.tight_layout()
    plt.savefig("scatter_plot.png")
    plt.show()
    plt.close()


def prepare_row_cache(
    ds: Dataset, features: list[str]
) -> tuple[list[str | None], dict[str, list[float | None]]]:
    """
    Cache parsed floats once for all features

    house_by_row: the house names aligned with ds.rows (list[str|None])
    values_by_feature:
    Dict: feature_name(str) -> list aligned with ds.rows (list[float|None])
    Each element is float (valid) or None (empty/invalid)
    """
    house_by_row: list[str | None] = []
    values_by_feature: dict[str, list[float | None]] = {}
    for name in features:
        values_by_feature[name] = []
    label_idx = None
    if ds.label_name is not None:
        try:
            label_idx = ds.header.index(ds.label_name)
        except Exception:
            label_idx = None
    feature_idx: dict[str, int] = {}
    for name in features:
        feature_idx[name] = ds.header.index(name)
    for row in ds.rows:
        if label_idx is None:
            house_by_row.append(row)
        else:
            housename = row[label_idx].strip()
            if housename:
                house_by_row.append(housename)
            else:
                house_by_row.append(None)
        for name in features:
            idx = feature_idx[name]
            values_by_feature[name].append(get_float(row[idx]))
    return (house_by_row, values_by_feature)


def main() -> int:
    """
    main function
    """
    try:
        dataset_path: str | Path = get_csv_path()
        ds = load_dataset(dataset_path)
        assert ds.rows, "empty dataset"
        exclude: list[str] = []
        if ds.index_name:
            exclude.append(ds.index_name)
        if ds.label_name:
            exclude.append(ds.label_name)
        # print(f"index_name is {ds.index_name},
        # label_name is {ds.label_name}")
        features = column_has_number(ds, exclude=exclude)
        assert features, "no numeric features found"
        best = choose_most_similiar_pair(ds, features)
        assert best is not None, "could not find a valid feature pair"
        x_name, y_name, r, n_points = best
        houses = get_house_names(ds)

        print(f"Selected pair: {x_name} vs {y_name}")
        print(
            f"Pearson correlation coefficient r: {r:.6f}, |\
r|: {abs(r):.6f}, n: {n_points}"
        )
        if not houses:
            print(
                "Info: no house labels found, plotting without house colours"
            )
        scatter_plot(ds, houses, x_name, y_name, r)
        return 0
    except Exception as e:
        print(f"AssertionError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    """
    Entrypoint
    """
    raise SystemExit(main())
