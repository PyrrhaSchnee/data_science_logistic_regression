#!/usr/bin/env python3
"""
logreg_train.py

Train a one-vs-all logistic regression model using gradient descent.
Store result in after_train.csv which contains:
- feature list (header)
- mean and std for scaling
- theta per house
------------------------------------------
Flow of process
1) load the csv into a Dataset
input file: dataset_train.csv
Dataset structure:
ds.header: list of column names
ds.rows: list of rows, each row is a list of strings
ds.label_name: "Hogwarts House"
ds.index_name: "Index"
We need both the features (grades, the numbers) and the label (housename)
for training


2) Determine which columns are numeric (which columns contains numbers)
build "feature_names": list of course columns (like Arithmancy, Astronomy ...)
exclude ds.label_name and ds.index_name because not a feature (not a number)
Output:
"feature_names": list[str] = ["Arithmancy", "Astronomy", ...]
"feature_count": int = len(feature_names) (in our case, it should be 13)
Logistic regression needs numeric input vectors.
Here vector means a list of number.
For example, a student has 100 for math, 98 for english, 99 for chemistry,
then the vector representating his grades are like:
["100", "98", "99"]
with the feature_names = ["math", "english", "chemistry"]
and feature_count = len(feature_names) = 3


3) Extract raw training data (X raw + labels y)
Note: "X raw + labels y" is the standard machine learning notation:
X = the inputs/features (here the numbers)
It is in capital letter because it represents all of the students (X[i])
y = the labels/targets (here the housename)
It is not capital letter because there is only one single label (housename)
per student
X and y are aligned (same number of samples, not same number items)
3.1) raw_x_rows: list[list[float|None]]
a list of samples, each sample is a list of length len(feature_count),
each entry is either a float number (valid) or None (empty/invalid)
example: raw_x_rows[0]:list[float|None] = [58384.0, -487.88, None, 4.87, ...]
3.2) y_labels:list[str]
a list of housename strings aligned with rows
example: y_labels[0]:str = "Ravenclaw"


4) Compute feature means
For each feature column j: sum only values that are not None, count only these
values, never replace them by zero, compute mean_j = sum_j / count_j
Output: feature_means:list[float]
feature_count:int = len(feature_means)
We need this step to be able to standardize, and to impute emtpy/invalid values


5) Impute empty/invalid values with the result of step 4
For every sample i, feature j:
if raw_x_rows[i][j] is None, then replace it by feature_means[j]
Output:
imputed_x_rows:list[list[float]], no None inside
We need this step because gradient descent only accepts numbers, not None,
and prediction also only works with numbers internal, not None


6) Compute the standard deviations of the features (stds)
After step 5 we guarantee every feature j has only numbers
(thanks to imputation)
Now we compute:
stdj = sqrt((1 / len(imputed_x_rows)) * sum of (xij - meanj)**2 )
xij is the grade of student i in course j.
if stdj = 0.0, set it to 1.0 to avoid division by zero
Output: feature_stds:list[float]
len(feature_stds) = feature_count
We need this step because standardization makes training stable and fast


7) Standardize features
For each sample i, feature j,
x'ij = (xij - meanj) / stdj
Output:
x_std_rows:list[float]
We need this step because it prevents one "large scale" features from
dominating other gradients


8) Intercept (add bias term)
Convert each sample from [x1, x2, x3, ..., xn] to [1.0, x1, x2, x3 ..., xn]
Output:
x_rows:list[float], this is what will be used for training
len(x_rows) = feature_count + 1
We need this step because it allows the model to learn an intercept theta0
which is a constant number

9) Train one-vs-all models (4 binary models in total)
Retrieve housenames: houses:list[str] = get_house_names(ds)
For each item houses[i] in houses:
9.1) Build binary target vector y_bin:
for each sample i:
y_bin[i] = 1.0 if y_labels[i] == houses[i]
else y_bin[i] = 0.0
9.2) Run gradient descent to learn theta
Initialization : theta = [0.0] * (feature_count + 1)
For each epoch:
compute predictions:
pred_i = sigmoid(theta · x_rows[i])
compute gradient loss wrt theta, update theta
Output: one theta per house
Store them: theta_by_house:dict[str, float]
theta_by_house[houese[i]] = theta


10) Save the result after training for the prediction
save :
feature_names
feature_means
feature_stds
theta_by_house
Output: after_train.csv
"""


from __future__ import annotations
import csv
import math
import random
import sys
from pathlib import Path
from utilities import (
    Dataset,
    column_has_number,
    get_float,
    load_dataset,
    get_house_names,
    get_csv_path,
    dot_product,
)

RESULT_FILE = "after_train.csv"
LEARNING_RATE = 0.1
EPOCH_COUNT = 1500
"""
# Mandatory : batch GD, Bonus = SGD and mini-batch
# - btach GD : BATCH_SIZE = 0 (use all samples)
# - Stochiastic GD : BATCH_SIZE = 1
# - mini-batch : BATCH_SIZE = 32 or 64
"""
BATCH_SIZE = 0
"""
# Bonus: momentum beta
# beta is to used to reduce zig-zag behavior while
# iterating, because more often than not, the good result
# is in one certain direction which is observed/observable
# on most steps, but not on all steps. When we arrive at
# those fraction of steps that the persuaded good direction
# is not aligned the initial global tendence, we waste time
# to get out of it and return to the good trajectory.
# momentum beta allows us to assume and enforce the above hypothesis
# so we change direction only when there has been a certain number
# of consecutive steps that we think we are in the wrong directionl
# set it to 0.9 to activate it, set it to 0.0 to disable it
"""
MOMENTUM_BETA = 0.0


def sigmoid(z: float) -> float:
    """
    Numerically stable sigmoid
    """
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        return math.exp(z) / (1.0 + math.exp(z))


def build_feature_matrix(
    ds: Dataset, feature_names: list[str]
) -> tuple[list[list[float | None]], list[str]]:
    """
    Extract raw features and labels from dataset
    - raw_x_rows: list[list[float | None]]
    list of samples, each sample contains list of float
    or None(invalid/empty)
    - y_labels: list[str]
    list of housenames as strings (must be present)
    """
    assert ds.label_name is not None, "housename must be present"
    raw_x_rows: list[list[float | None]] = []
    y_labels: list[str] = []
    label_idx = ds.header.index(ds.label_name)
    feature_idx: list[int] = []
    for name in feature_names:
        feature_idx.append(ds.header.index(name))
    feature_count = len(feature_idx)
    for row in ds.rows:
        housename = row[label_idx].strip()
        if not housename:
            continue
        one_student: list[float | None] = []
        j = 0
        while j < feature_count:
            value = get_float(row[feature_idx[j]])
            one_student.append(value)
            j += 1
        raw_x_rows.append(one_student)
        y_labels.append(housename)
    return (raw_x_rows, y_labels)


def compute_means(raw_x_rows: list[list[float | None]]) -> list[float]:
    """
    compute per-feature mean, ignoring None
    this mean will be used for both imputation and scaling
    """
    feature_count = len(raw_x_rows[0])
    sums = [0.0] * feature_count
    counts = [0] * feature_count
    i = 0
    while i < len(raw_x_rows):
        j = 0
        while j < feature_count:
            value = raw_x_rows[i][j]
            if value is not None:
                sums[j] += value
                counts[j] += 1
            j += 1
        i += 1
    means: list[float] = []
    j = 0
    while j < feature_count:
        if counts[j] == 0:
            means.append(0.0)
        else:
            means.append(sums[j] / float(counts[j]))
        j += 1
    return means


def impute_and_standardize(
    raw_x_rows: list[list[float | None]], feature_means: list[float]
) -> tuple[list[list[float]], list[float]]:
    """
    1) Replace None by feature mean (mean imputation)
    2) Compute std after imputation
    3) Standarize: (x - mean) / std

    Return:
    - x_rows: standarized data (no more None), without bias term
        (we did mean imputation)
    - feature_stds: per-feature std (never zero)
    """
    sample_count = len(raw_x_rows)
    feature_count = len(feature_means)
    imputed: list[list[float]] = []
    i = 0
    while i < sample_count:
        j = 0
        row: list[float] = []
        while j < feature_count:
            value = raw_x_rows[i][j]
            if value is not None:
                row.append(value)
            else:
                row.append(feature_means[j])
            j += 1
        imputed.append(row)
        i += 1
    feature_stds: list[float] = []
    j = 0
    while j < feature_count:
        mu = feature_means[j]
        var = 0.0
        i = 0
        while i < sample_count:
            d = imputed[i][j] - mu
            var += d * d
            i += 1
        var /= float(sample_count)
        std = math.sqrt(var)
        if std == 0.0:
            std = 1.0
        feature_stds.append(std)
        j += 1
    x_rows: list[list[float]] = []
    i = 0
    while i < sample_count:
        row2: list[float] = []
        j = 0
        while j < feature_count:
            row2.append((imputed[i][j] - feature_means[j]) / feature_stds[j])
            j += 1
        x_rows.append(row2)
        i += 1
    return (x_rows, feature_stds)


def add_intercept(x_rows: list[list[float]]) -> list[list[float]]:
    """
    add 1.0 to the 1st position of each sample
    which represents the intercept term theta0
    """
    ret: list[list[float]] = []
    i = 0
    while i < len(x_rows):
        ret.append([1.0] + x_rows[i])
        i += 1
    return ret


def train_once(
    x_rows: list[list[float]],
    y_bin: list[float],
    batch_size: int,
    momentum_beta: float,
    learning_rate: float,
    epoch_count: int,
) -> list[float]:
    """
    Train the binary logistic regression model
    """
    sample_count = len(x_rows)
    feature_count = len(x_rows[0])
    theta = [0.0] * feature_count
    velocity = [0.0] * feature_count
    if batch_size <= 0 or batch_size > sample_count:
        batch_size = sample_count
    idx = list(range(sample_count))
    epoch = 0
    while epoch < epoch_count:
        if batch_size != sample_count:
            random.shuffle(idx)
        start = 0
        while start < sample_count:
            end = start + batch_size
            if end > sample_count:
                end = sample_count
            grad = [0.0] * feature_count
            batch_len = end - start
            k = start
            while k < end:
                i = idx[k]
                p = sigmoid(dot_product(theta, x_rows[i]))
                err = p - y_bin[i]
                j = 0
                while j < feature_count:
                    grad[j] += err * x_rows[i][j]
                    j += 1
                k += 1
            j = 0
            while j < feature_count:
                g = grad[j] / float(batch_len)
                if momentum_beta > 0.0:
                    velocity[j] = momentum_beta * velocity[j] + g
                    theta[j] -= learning_rate * velocity[j]
                else:
                    theta[j] -= learning_rate * g
                j += 1
            start = end
        epoch += 1
    return theta


def train_all(
    x_rows: list[list[float]], y_labels: list[str], houses: list[str]
) -> dict[str, list[float]]:
    """
    Train one model per house
    """
    theta_by_house: dict[str, list[float]] = {}
    house_idx = 0
    while house_idx < len(houses):
        target = houses[house_idx]
        y_bin = []
        i = 0
        while i < len(y_labels):
            if y_labels[i] == target:
                y_bin.append(1.0)
            else:
                y_bin.append(0.0)
            i += 1
        theta = train_once(
            x_rows=x_rows,
            y_bin=y_bin,
            batch_size=BATCH_SIZE,
            momentum_beta=MOMENTUM_BETA,
            learning_rate=LEARNING_RATE,
            epoch_count=EPOCH_COUNT,
        )
        theta_by_house[target] = theta
        house_idx += 1
    return theta_by_house


def predict(x: list[float], theta_by_house: dict[str, list[float]]) -> str:
    """
    compute the score for all the houses and pick
    that one with the highest score
    We compare raw z = theta · x because sigmoid is monotonic
    """
    best_house: str = ""
    best_score: float | None = None
    for house, theta in theta_by_house.items():
        score = dot_product(theta, x)
        if best_score is None or best_score < score:
            best_score = score
            best_house = house
    return best_house


def compute_training_accuracy(
    x_rows: list[list[float]],
    y_labels: list[str],
    theta_by_house: dict[str, list[float]],
) -> float:
    """
    Compute training accurancy
    """
    value = 0
    i = 0
    while i < len(x_rows):
        pred = predict(x_rows[i], theta_by_house)
        if pred == y_labels[i]:
            value += 1
        i += 1
    return value / float(len(x_rows))


def save_training(
    output: Path,
    feature_names: list[str],
    feature_means: list[float],
    feature_stds: list[float],
    theta_by_house: dict[str, list[float]],
) -> None:
    """
    Save the training result into a csv file
    """
    with output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, lineterminator="\n")
        writer.writerow(["type", "house", "theta0"] + feature_names)
        writer.writerow(
            ["mean", "", "0.0"] + [f"{x:.15f}" for x in feature_means]
        )
        writer.writerow(
            ["std", "", "0.0"] + [f"{x:.15f}" for x in feature_stds]
        )
        for house in sorted(theta_by_house.keys()):
            theta = theta_by_house[house]
            row = ["theta", house] + [f"{x:.15f}" for x in theta]
            writer.writerow(row)


def main() -> int:
    """
    main function
    """
    try:
        dataset_path: str | Path = get_csv_path()
        ds = load_dataset(dataset_path)
        assert ds.rows, "empty dataset"
        houses = get_house_names(ds)
        assert houses, "No housename found in the dataset"
        exclude: list[str] = []
        if ds.index_name:
            exclude.append(ds.index_name)
        if ds.label_name:
            exclude.append(ds.label_name)
        feature_names = column_has_number(ds, exclude=exclude)
        assert feature_names, "No nuemric features found in the dataset"
        raw_x_rows, y_labels = build_feature_matrix(ds, feature_names)
        feature_means = compute_means(raw_x_rows)
        x_std, feature_stds = impute_and_standardize(raw_x_rows, feature_means)
        x_rows = add_intercept(x_std)
        theta_by_house = train_all(x_rows, y_labels, houses)
        accuracy = compute_training_accuracy(x_rows, y_labels, theta_by_house)
        print(f"Train accuracy is: {accuracy:.6f}")
        save_training(
            Path(RESULT_FILE),
            feature_names,
            feature_means,
            feature_stds,
            theta_by_house,
        )
        print(f"Training result saved to {RESULT_FILE}")
        return 0
    except Exception as e:
        print(f"AssertionError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    """
    Entrypoint
    """
    raise SystemExit(main())
