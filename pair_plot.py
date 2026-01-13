#!/usr/bin/env python3
"""
pair_plot.py

scatter plot matrix visualization
"""


from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import sys
from utilities import (
    Dataset,
    get_csv_path,
    load_dataset,
    column_has_number,
    get_house_names,
)
from scatter_plot import prepare_row_cache


def global_view(
    features: list[str],
    houses: list[str],
    house_by_row: list[str | None],
    values_by_feature: dict[str, list[float | None]],
) -> None:
    """
    Generate the global view which is all the plotted images
    combined in a single file

    layout:
    - row_idx and col_idx are subplot coordinates in the grid
    - x_name = features[col_idx] is the feature on X-axis
    - y_name = features[row_idx] is the feature on Y-axis

    only half of all need to be displayed to avoid duplication
    """
    output_path = Path.cwd() / "scatter_global_view.png"
    n = len(features)
    figure, axes = plt.subplots(n, n, figsize=(1.8 * n, 1.8 * n))
    row_idx = 0
    while row_idx < n:
        col_idx = 0
        while col_idx < n:
            ax = axes[row_idx][col_idx]
            y_name = features[row_idx]
            x_name = features[col_idx]
            if row_idx > col_idx:
                ax.axis("off")
            elif row_idx == col_idx:
                ax.text(
                    0.5,
                    0.5,
                    x_name,
                    ha="center",
                    va="center",
                    fontsize=9,
                    wrap=True,
                )
            else:
                x_vals = values_by_feature[x_name]
                y_vals = values_by_feature[y_name]
                if houses:
                    for housename in houses:
                        xs: list[float] = []
                        ys: list[float] = []
                        for rh, x, y in zip(house_by_row, x_vals, y_vals):
                            if rh != housename or y is None or x is None:
                                continue
                            xs.append(x)
                            ys.append(y)
                        ax.scatter(xs, ys, s=4, alpha=0.5)
                else:
                    xs2: list[float] = []
                    ys2: list[float] = []
                    for x, y in zip(x_vals, y_vals):
                        if x is None or y is None:
                            continue
                        xs2.append(x)
                        ys2.append(y)
                    ax.scatter(xs2, ys2, s=4, alpha=0.5)
            col_idx += 1
        row_idx += 1
    figure.suptitle("Global view of all scatter plots", fontsize=14, y=0.995)
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    figure.savefig(output_path, dpi=350)
    plt.show()
    plt.close()


def main() -> int:
    """
    main function
    """
    try:
        dataset_path: str | Path = get_csv_path()
        ds: Dataset = load_dataset(dataset_path)
        assert ds.rows, "empty dataset"
        exclude: list[str] = []
        if ds.index_name:
            exclude.append(ds.index_name)
        if ds.label_name:
            exclude.append(ds.label_name)
        features = column_has_number(ds, exclude=exclude)
        assert features, "no numeric features found"
        house_by_row, values_by_feature = prepare_row_cache(ds, features)
        houses = get_house_names(ds)
        global_view(features, houses, house_by_row, values_by_feature)
        return 0
    except Exception as e:
        print(f"AssertionError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    """
    Entrypoint
    """
    raise SystemExit(main())
