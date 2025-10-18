import argparse
import os
import random

import pandas as pd


def parse_arg():
    parser = argparse.ArgumentParser(
        description="Take a dataset as input and analyse it"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        default="data/data_with_headers.csv",
        help="Path to dataset"
    )
    parser.add_argument(
        dest="test_size",
        type=float,
        default=0.2,
        help="A float between 0.0 and 1.0 representing the percentage for test portion"
    )
    return parser.parse_args()


def train_test_split(input_x: pd.DataFrame, input_y: pd.Series, test_size: float = 0.2, shuffle: bool = False, random_state: int = None) -> tuple:
    """
    A function that split the input dataset into X_train, X_test, y_train, y_test

    Args:
        input_x: pd.DataFrame of features data
        input_y: pd.Series of target column
        test_size: portion of test sets (< 1 and > 0)
        shuffle: bool of whether to randomly select rows from inputs
        random_state: int seed for reproducible random splits
    """
    size_X = len(input_x.index)
    size_y = len(input_y.index)
    if size_X != size_y:
        raise ValueError("input X and y have different sample size")
    if test_size >= 1 or test_size <= 0:
        raise ValueError(f"test size {test_size} is not in range (0.0, 1.0)")
    size_test = int(size_X * test_size)
    size_train = size_X - size_test

    if not shuffle:
        return input_x.iloc[:size_train], input_x.iloc[size_train:], input_y[:size_train], input_y[size_train:]

    # Set random seed for reproducible results
    if random_state is not None:
        random.seed(random_state)

    index_r = list(range(0, size_X))
    random.shuffle(index_r)
    # print(index_r[:size_test])
    # print(index_r[size_test:])

    # user dataframe.loc[list_of_rows, list_of_columns] to select
    return input_x.loc[index_r[size_test:]], input_x.loc[index_r[:size_test]], input_y.loc[index_r[size_test:]], input_y.loc[index_r[:size_test]]


def columns_to_exclude(target_cols: list, input_col: str) -> bool:
    """
    Check if input column is in the target columns.
    """
    for target in target_cols:
        if input_col.startswith(target):
            return True
    return False


def main():
    args = parse_arg()
    # input checks
    data_path = args.dataset
    if not os.path.exists(data_path) or not os.path.isfile(data_path) or not data_path.endswith(".csv"):
        raise Exception("Dataset csv does not exist or not a csv file.")
    test_size = args.test_size
    if test_size <= 0 or test_size >= 1:
        raise Exception("Test size must be a float between 0 and 1.")

    # read csv
    data = pd.read_csv(data_path)
    # feature selection
    ditch_columns = ["ID", "Diagnosis", "smoothness", "symmetry", "fractal_dim"]
    selected_cols = [col for col in data.columns.tolist() if not columns_to_exclude(ditch_columns, col)]
    # print(selected_cols)
    y = data["Diagnosis"]
    X = data[selected_cols]
    print(type(y))
    print(type(X))

    # call train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, shuffle=True)

    # save split data
    data_train = X_train
    data_train["Diagnosis"] = y_train
    data_train.to_csv("data/data_train.csv", index=False)
    data_test = X_test
    data_test["Diagnosis"] = y_test
    data_test.to_csv("data/data_test.csv", index=False)


if __name__ == "__main__":
    main()
