import argparse
import json

import numpy as np
import pandas as pd

from src.MLP.models import MLP, DenseConfig


def parse_arg():
    parser = argparse.ArgumentParser(
        description="Take path to training dataset for MLP training."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        default="data/data_train.csv",
        help="Path to training dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nn_config.json",
        help="path to neural network configuration"
    )
    return parser.parse_args()


def nn_config_gen(config_path: str, feat_size: int) -> list[DenseConfig]:
    """
    Args:
        feat_size: number of features for input data
        config_path: path to a json file including nn configs
    return:
        a list of DenseConfig for MLP model to init Dense Layers
    """
    output_confs = []

    # Parse JSON configuration file
    with open(config_path) as f:
        nn_confs = json.load(f)
        # print(nn_confs)

    input_shape = feat_size

    for layer in nn_confs:
        cfg = DenseConfig(
            input_shape=input_shape,
            output_shape=layer["nodes"],
            activation=layer["activation"],
            weights_init=layer["weights_init"]
        )
        output_confs.append(cfg)
        input_shape = layer["nodes"]

    return output_confs


def zscore(x):
    return (x - np.mean(x)) / np.std(x)


def main():
    args = parse_arg()
    data = pd.read_csv(str(args.dataset))
    # 1. tranfrom binary result of Diagnosis: B = 0, M = 1
    # generate binary result (df_one) of B and M
    df_one = pd.get_dummies(data["Diagnosis"], dtype=int)
    # print(df_one)
    # add df_one to dataframe and replace Dianosis with the result of M == True
    data = pd.concat((data, df_one), axis=1)
    data = data.drop(["Diagnosis", "B"], axis=1)
    data = data.rename(columns={"M": "Diagnosis"})
    y_train = data["Diagnosis"]
    x_train = data.drop(["Diagnosis"], axis=1)
    # print(y_train)
    # print(x_train)
    # print(x_train.shape)

    # 2. apply zscore normalization to each feature
    features = list(x_train.columns)
    for feat in features:
        x_train[feat] = zscore(x_train[feat])

    # Convert pandas DataFrames to numpy arrays for neural network processing
    x_train = x_train.values
    y_train = y_train.values.reshape(-1, 1)  # Reshape to column vector for consistency

    # Generate network configuration from JSON
    network_config = nn_config_gen(str(args.config), x_train.shape[1])
    # print(network_config)

    # Init MLP network instance
    mlp = MLP(network_config)
    print(mlp)

    # Train with input data
    mlp.fit(x_train, y_train)


if __name__ == "__main__":
    main()
