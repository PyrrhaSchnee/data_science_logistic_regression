#!/usr/bin/env python3
"""
logreg_predict.py

Predict which house is the subject most likely to belong to
"""

from __future__ import annotations
import sys
import csv
from pathlib import Path
from utilities import Dataset, get_float, load_dataset, dot_product

OUTPUT_FILE = "houses.csv"
TRAINING_RESULT = "after_train.csv"
TEST_FILE = "dataset_test.csv"


def load_training_result(
    pathname: str,
) -> tuple[list[str], list[float], list[float], dict[str, list[float]]]:
    """
    Load data from after_train.csv then return:
    - feature_names
    - feature_means
    - feature_stds
    - theta_by_house
    """
    path: Path = Path(pathname)
    with path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        assert header is not None and len(header) >= 4, "Invalid training data"
        feature_names = header[3:]
        feature_means: list[float] | None = None
        feature_stds: list[float] | None = None
        theta_by_house: dict[str, list[float]] = {}
        for row in reader:
            if not row:
                continue
            row_type = row[0].strip()
            if row_type == "mean":
                feature_means = [float(x) for x in row[3:]]
            elif row_type == "std":
                feature_stds = [float(x) for x in row[3:]]
            elif row_type == "theta":
                house = row[1].strip()
                theta = [float(x) for x in row[2:]]
                theta_by_house[house] = theta
    assert (
        feature_means is not None
        and feature_stds is not None
        and theta_by_house
    ), "Invalid training data"
    return (feature_names, feature_means, feature_stds, theta_by_house)


def standardize_sample(
    raw_values: list[float | None], means: list[float], stds: list[float]
) -> list[float]:
    """
    Impute missing values with means then standardize then add intercept
    Returns: [1.0, x1', x2', ...]
    """
    ret = [1.0]
    i = 0
    while i < len(raw_values):
        value = raw_values[i]
        if value is None:
            value = means[i]
        ret.append((value - means[i]) / stds[i])  # type: ignore
        i += 1
    return ret


def predict_house(
    x: list[float], theta_by_house: dict[str, list[float]]
) -> str:
    """
    predict which house it belongs to by computing the highest theta·x score
    """
    best_house: str = ""
    best_score: float | None = None
    for house, theta in theta_by_house.items():
        score = dot_product(theta, x)
        if best_score is None or score > best_score:
            best_score = score
            best_house = house
    return best_house


def get_csv_path_2(pathname: str) -> Path | str:
    """
    Take the user input path
    """
    argv = sys.argv
    path = Path(pathname)
    if len(argv) == 1:
        return path
    if len(argv) == 2:
        p = argv[1]
        assert p.lower().endswith(".csv"), "Input must be a .csv file"
        return p
    raise ValueError("Usage: %s [xxxx.csv]" % (argv[0],))


def main() -> int:
    """
    main function
    """
    try:
        dataset_path: str | Path = get_csv_path_2(TEST_FILE)
        ds: Dataset = load_dataset(dataset_path)
        assert ds.rows, "empty dataset"
        feature_names, means, stds, theta_by_house = load_training_result(
            TRAINING_RESULT
        )
        assert ds.index_name, "Dataset must have an index column"
        index_idx = ds.header.index(ds.index_name)
        feature_idx: list[int] = []
        for name in feature_names:
            feature_idx.append(ds.header.index(name))
        output: Path = Path(OUTPUT_FILE)
        with output.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, lineterminator="\n")
            writer.writerow(["Index", "Hogwarts House"])
            for row in ds.rows:
                idx = row[index_idx].strip()
                raw_values: list[float | None] = []
                i = 0
                while i < len(feature_idx):
                    raw_values.append(get_float(row[feature_idx[i]]))
                    i += 1
                x = standardize_sample(raw_values, means, stds)
                house = predict_house(x, theta_by_house)
                writer.writerow([idx, house])
        print(f"Prediction result has been saved to {OUTPUT_FILE}")
        return 0
    except Exception as e:
        print(f"AssertionError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    """
    Entrypoint
    """
    raise SystemExit(main())
