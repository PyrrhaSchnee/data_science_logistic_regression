#!/usr/bin/env python3
"""
histogram.py

Displays a histogram in order to answer the following question:
Which Hogwarts course has a homogeneous score distribution between \
all for houses ?
"""

from __future__ import annotations
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


def extract_values_by_house(
    ds: Dataset, houses: list[str], feature: str
) -> dict[str, list[float]]:
    """
    Get values from different houses
    """
    house_col = ds.label_name
    if house_col is None:
        return {}
    try:
        feature_idx = ds.header.index(feature)
        house_idx = ds.header.index(house_col)
    except Exception:
        return {}
    values: dict[str, list[float]] = {house_name: [] for house_name in houses}
    for row in ds.rows:
        house = row[house_idx].strip()
        if house not in values:
            continue
        x = get_float(row[feature_idx])
        if x is None:
            continue
        values[house].append(x)
    return values


def get_min_and_max(
    values_by_house: dict[str, list[float]], houses: list[str]
) -> tuple[float, float] | None:
    min_v = None
    max_v = None
    for house in houses:
        for x in values_by_house.get(house, []):
            if min_v is None or x < min_v:
                min_v = x
            if max_v is None or x > max_v:
                max_v = x
    if min_v is None or max_v is None:
        return None
    return (min_v, max_v)


def normalize_histogram(
    values: list[float], min_v: float, max_v: float, bins: int
) -> list[float]:
    """
    Build a normalized histogram (probability vector) with fixed bins
    over [min_v, max_v],
    The output sums to 1.0 then data is not void / invalid
    (discrete probability density)
    """
    if bins <= 0:
        return []
    if not values:
        return [0.0] * bins
    if min_v == max_v:
        ret = [0.0] * bins
        ret[bins // 2] = 1.0
        return ret
    width = (max_v - min_v) / float(bins)
    counts = [0] * bins
    total = 0
    for x in values:
        idx = int((x - min_v) / width)
        if idx < 0:
            idx = 0
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
        total += 1

    if total == 0:
        return [0.0] * bins
    return [x / float(total) for x in counts]


def get_average_pairwise_l1(
    houses: list[str], probs_by_house: dict[str, list[float]]
) -> float:
    """
    Compute the average pairwise L1 distance between houses'
    normalized histograms
    Smaller value means the distributions are more homogeneous
    """
    distance_sum: float = 0.0
    pairs: int = 0
    i: int = 0
    while i < len(houses):
        j: int = i + 1
        while j < len(houses):
            a = probs_by_house[houses[i]]
            b = probs_by_house[houses[j]]
            k: int = 0
            d: float = 0.0
            while k < len(a) and k < len(b):
                d += abs(a[k] - b[k])
                k += 1
            distance_sum += d
            pairs += 1
            j += 1
        i += 1
    if pairs == 0:
        return 0.0
    return distance_sum / float(pairs)


def choose_best_feature(
    ds: Dataset, houses: list[str], features: list[str], bins: int
) -> tuple[str, float] | None:
    """
    Select the feature which per-house histogram distrubtions
    are the most homogeneous using average pairwise L1 distance as the metric
    """
    best_name: str | None = None
    best_score: float | None = None
    for name in features:
        values_by_house = extract_values_by_house(ds, houses, name)
        min_max = get_min_and_max(values_by_house, houses)
        if min_max is None:
            continue
        min_v, max_v = min_max
        probs: dict[str, list[float]] = {}
        for h in houses:
            probs[h] = normalize_histogram(
                values_by_house.get(h, []), min_v, max_v, bins
            )
        score: float | None = get_average_pairwise_l1(houses, probs)
        if best_score is None or best_score > score:
            best_score = score
            best_name = name
    if best_score is None or best_name is None:
        return None
    return (best_name, best_score)


def plot_feature_histogram(
    ds: Dataset, houses: list[str], feature: str, bins: int
) -> None:
    """
    Plot overlaid normalized histograms for the selected feature, one per house
    using identical bin edges for reliable visual assessment
    """
    values_by_house = extract_values_by_house(ds, houses, feature)
    min_max = get_min_and_max(values_by_house, houses)
    if min_max is None:
        raise AssertionError("No numeric data to be plotted")
    min_v, max_v = min_max
    if min_v == max_v:
        edges = [min_v - 0.5, max_v + 0.5]
    else:
        step = (max_v - min_v) / float(bins)
        edges = []
        i = 0
        while i <= bins:
            edges.append(min_v + step * float(i))
            i += 1

    plt.figure()
    for h in houses:
        v = values_by_house.get(h, [])
        plt.hist(v, bins=edges, alpha=0.5, density=True, label=h)

    plt.title(f"Most homogeneous course is: {feature}")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("histogram.png")
    plt.show()
    plt.close()


def main() -> int:
    """
    Load the .csv file to generate Dataset, retrieve house
    names and numeric features, find the most homogeneous one
    and print the result and plot the histogram
    """
    try:
        dataset_path: str | Path = get_csv_path()
        ds = load_dataset(dataset_path)
        assert ds, "Empty dataset detected"
        houses = get_house_names(ds)
        exclude = []
        if ds.index_name:
            exclude.append(ds.index_name)
        if ds.label_name:
            exclude.append(ds.label_name)
        features = column_has_number(ds, exclude=exclude)
        assert features, "No numeric features found"
        bins = 25
        best = choose_best_feature(ds, houses, features, bins)
        assert best, "Could not determine the most homogeneous course"
        best_name, best_score = best
        # debug_feature(ds, houses, best_name, bins)
        print(
            f"Selected feature: {best_name} (homogeneity \
score = {best_score:.6f})"
        )
        plot_feature_histogram(ds, houses, best_name, bins)
        return 0
    except Exception as e:
        print(f"AssertionError: {e}", file=sys.stderr)
        return 1


def debug_feature(
    ds: Dataset, houses: list[str], feature: str, bins: int
) -> None:
    """
    Debugger
    Print per-house counts, histogram sums and pairwise L1 distance
    for a feature.
    Run this when result is suspicious
    """
    values_by_house = extract_values_by_house(ds, houses, feature)
    min_max = get_min_and_max(values_by_house, houses)
    assert min_max is not None, "DEBUG: no data"
    min_v, max_v = min_max
    print(f"debug: feature={feature} bins={bins} mn={min_v} mx={max_v}")
    probs: dict[str, list[float]] = {}
    for h in houses:
        vals = values_by_house.get(h, [])
        p = normalize_histogram(vals, min_v, max_v, bins)
        probs[h] = p
        print(
            f"debug: {h}: n={len(vals)} \
sum={sum(p):.6f} maxbin={max(p) if p else 0.0:.6f}"
        )
    i = 0
    while i < len(houses):
        j = i + 1
        while j < len(houses):
            a = probs[houses[i]]
            b = probs[houses[j]]
            d = 0.0
            k = 0
            while k < len(a) and k < len(b):
                d += abs(a[k] - b[k])
                k += 1
            print(f"debug: L1({houses[i]},{houses[j]})={d:.6f}")
            j += 1
        i += 1

    print(
        f"debug: avgerage_pairwise_L1=\
{get_average_pairwise_l1(houses, probs):.6f}"
    )


if __name__ == "__main__":
    """
    Entrypoint
    """
    raise SystemExit(main())
