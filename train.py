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
    # 1. Create one-hot encoded labels for multi-class classification
    # Generate one-hot encoded result (df_one) of B and M
    df_one = pd.get_dummies(data["Diagnosis"], dtype=int)
    # print(df_one)

    # Remove the original Diagnosis column and keep the one-hot encoded columns
    data = data.drop(["Diagnosis"], axis=1)

    # Separate features and labels
    x_train = data
    y_train = df_one  # Keep both B and M columns for one-hot encoding

    # Ensure columns are in consistent order: [B, M]
    if 'B' in y_train.columns and 'M' in y_train.columns:
        y_train = y_train[['B', 'M']]  # B=0, M=1 becomes [1,0] for B, [0,1] for M

    # print(y_train)
    # print(x_train)
    # print(x_train.shape)

    # 2. apply zscore normalization to each feature
    features = list(x_train.columns)
    for feat in features:
        x_train[feat] = zscore(x_train[feat])

    # Convert pandas DataFrames to numpy arrays for neural network processing
    x_train = x_train.values
    y_train = y_train.values  # Keep as 2D array for one-hot encoded labels

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
