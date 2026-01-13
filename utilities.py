#!/usr/bin/env python3
"""
utilities.py

utilities
"""

from __future__ import annotations
import math
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TRAIN_FILE = "dataset_train.csv"
TEST_FILE = "dataset_test.csv"
MISSING_FIELD: set[str] = {"", "nan", "null", "none"}


@dataclass(frozen=True)
class Dataset:
    """
    Leverage dataclass to automatically create a struct-like class,
    which allows us to easily store and access
    data across the entire program.

    frozen=True makes the object immutable after creation
    This avoids accidental overwrites
    """

    header: list[str]
    rows: list[list[str]]
    index_name: str | None
    label_name: str | None


def dot_product(a: list[float], b: list[float]) -> float:
    """
    Dot product between vectors of same length
    """
    n = len(a)
    assert n == len(
        b
    ), "The 2 input vectors of dot_product must have the \
same length"
    i = 0
    ret = 0.0
    while i < n:
        ret += a[i] * b[i]
        i += 1
    return ret


def get_house_names(ds: Dataset) -> list[str]:
    """
    Construct the list of house names from Dataset
    Sort them then return the list
    """
    assert (
        ds.label_name is not None
    ), "Dataset has no column called 'Hogwarts House'"
    try:
        house_idx = ds.header.index(ds.label_name)
    except Exception as e:
        raise ValueError("Label column not found in the header") from e
    seen: set[str] = set()
    houses: list[str] = []
    for row in ds.rows:
        name = row[house_idx].strip()
        if not name:
            continue
        if name not in seen:
            seen.add(name)
            houses.append(name)

    houses.sort()
    if not houses:
        raise ValueError("No house names found")
    return houses


def get_csv_path() -> Path | str:
    """
    Take the user input path
    """
    argv = sys.argv
    path = Path(TRAIN_FILE)
    if len(argv) == 1:
        return path
    if len(argv) == 2:
        p = argv[1]
        assert p.lower().endswith(".csv"), "Input must be a .csv file"
        return p
    raise ValueError("Usage: %s [xxxx.csv]" % (argv[0],))


def get_float(s: str | None) -> float | None:
    """
    If a string can be converted to a float, returns it;
    otherwise, returns None.
    nan and inf are considered as invalid
    This avoids silently converting non-float to 0.0 which
    would silently corrupt the entire stats up
    """
    try:
        if s is None:
            return None
        ret = s.strip()
        if ret.lower() in MISSING_FIELD:
            return None
        ret = float(ret)
        if math.isnan(ret) or math.isinf(ret):
            return None
        return ret
    except Exception:
        return None


def get_25_quartile(sorted_v: list[float]) -> float:
    """
    get the 25% quartile
    """
    size: int = len(sorted_v)
    return float(sorted_v[int(0.25 * size)])


def get_75_quartile(sorted_v: list[float]) -> float:
    """
    get the 75% quartile
    """
    size: int = len(sorted_v)
    return float(sorted_v[int(0.75 * size)])


def get_50_quartile(sorted_v: list[float]) -> float:
    """
    get the 50% quartile
    """
    size: int = len(sorted_v)
    return float(sorted_v[int(0.5 * size)])


def get_median(sorted_v: list[float]) -> float:
    """
    get median of a sorted list of floats
    """
    size: int = len(sorted_v)
    if size == 0:
        return 0.0
    mid_index = size // 2
    if size % 2 == 1:
        return sorted_v[mid_index]
    return (sorted_v[mid_index - 1] + sorted_v[mid_index]) / 2.0


def get_mean(values: list[float]) -> float:
    """
    compute the mean of a list of float number
    """
    if not values:
        return 0.0
    total = 0.0
    size: int = 0
    for x in values:
        total += x
        size += 1
    if size == 0:
        return 0.0
    return total / size


def fill_column_with_number(ds: Dataset, col_name: str) -> list[float]:
    """
    Extract numbers from a column, handling missing values
    """
    try:
        i = ds.header.index(col_name)
    except Exception:
        return []
    ret: list[float] = []
    for row in ds.rows:
        temp = get_float(row[i])
        if temp is None:
            continue
        ret.append(temp)
    return ret


def column_has_number(ds: Dataset, exclude: Iterable[str] = ()) -> list[str]:
    """
    Check if the column has at least float value
    """
    exclude_set = {item for item in exclude}
    columns: list[str] = []
    for i, name in enumerate(ds.header):
        if name in exclude_set:
            continue
        has_number = False
        for row in ds.rows:
            temp = get_float(row[i])
            if temp is not None:
                has_number = True
                break
        if has_number:
            columns.append(name)
    return columns


def get_variance(sorted_v: list[float]) -> float:
    """
    get the variance
    """
    size: int = len(sorted_v)
    m = get_mean(sorted_v)
    return sum((item - m) ** 2 for item in sorted_v) / size


def get_std(sorted_v: list[float]) -> float:
    """
    get the standard deviation
    """
    return get_variance(sorted_v) ** 0.5


def load_dataset(path: str | Path) -> Dataset:
    """
    Load the dataset (that .csv file)
    Skip empty lines
    """
    if isinstance(path, str):
        csv_path = Path(path)
    elif isinstance(path, Path):
        csv_path = path
    assert (
        csv_path.exists() and csv_path.is_file()
    ), f"Dataset not found: {csv_path}"

    with csv_path.open(newline="", encoding="utf-8") as csvfile:
        # csv reader returns each row as a list of strings
        reader = csv.reader(csvfile)
        # doing next(reader, None) avoids crash if the input file is empty
        header = next(reader, None)
        assert header is not None, ".csv must not be empty"
        # if there is a header, convert it into a list[str] with its contents
        header = [h.strip() for h in header]
        rows: list[list[str]] = []
        for row in reader:
            # if not row, just skips it like empty line
            if not row:
                continue
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            elif len(row) > len(header):
                row = row[: len(header)]
            rows.append(row)

        # Tells if "index" and "hogwarts house" are present
        index_name = None
        label_name = None
        for name in header:
            if name.strip().lower() == "index":
                index_name = name
            if name.strip().lower() in ("hogwarts house", "hogwarts_house"):
                label_name = name
        return Dataset(
            header=header,
            rows=rows,
            index_name=index_name,
            label_name=label_name,
        )


def main() -> None:
    """
    Not directly executable
    """
    return None


if __name__ == "__main__":
    """
    Entrypoint
    """
    main()
