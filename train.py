import argparse
import json

import numpy as np
import pandas as pd

from src.MLP.models import MLP, DenseConfig
from train_test_split import columns_to_exclude, train_test_split


def parse_arg():
    parser = argparse.ArgumentParser(
        description="Take path to training dataset for MLP training."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        default="data/data_with_headers.csv",
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
    # Remove the original Diagnosis column and keep the one-hot encoded columns
    data = data.drop(["Diagnosis"], axis=1)

    # feature selection
    ditch_columns = ["ID", "Diagnosis", "smoothness", "symmetry", "fractal_dim"]
    selected_cols = [col for col in data.columns.tolist() if not columns_to_exclude(ditch_columns, col)]

    X = data[selected_cols]
    y = df_one
    # print(df_one)

    # Ensure columns are in consistent order: [B, M]
    if 'B' in y.columns and 'M' in y.columns:
        y = y[['B', 'M']]  # B=0, M=1 becomes [1,0] for B, [0,1] for M

    # train test split with fixed random seed for reproducible results
    x_train, x_test, y_train, y_test = train_test_split(X, y, 0.2, True, random_state=42)
    # print(x_train)
    # print(y_train)
    # print(x_test)
    # print(y_test)

    # 2. apply zscore normalization to each feature
    features = list(x_train.columns)
    for feat in features:
        x_train[feat] = zscore(x_train[feat])
        x_test[feat] = zscore(x_test[feat])

    # Convert pandas DataFrames to numpy arrays for neural network processing
    x_train = x_train.values
    y_train = y_train.values  # Keep as 2D array for one-hot encoded labels
    x_test = x_test.values
    y_test = y_test.values

    # Generate network configuration from JSON
    network_config = nn_config_gen(str(args.config), x_train.shape[1])
    # print(network_config)

    # Init MLP network instance with fixed random seed for reproducible results
    mlp = MLP(network_config, epoch=1000, lr=0.005, batch_size=32)
    print(mlp)

    # Train with input data
    mlp.fit(x_train, y_train)
    print(f"Loss after training is {mlp.train_loss}")

    # Predict with test data
    y_pred = mlp.predict(x_test)
    print("Sample predictions (first 5):")
    print(y_pred[:5])
    print("Sample ground truth (first 5):")
    print(y_test[:5])

    # Calculate validation loss
    loss = mlp.loss_function.loss(y_test, y_pred, eps=1e-15)
    print(f"Validation loss is {loss}")

    # Evaluate model performance
    evaluation = mlp.evaluate(y_pred, y_test)
    print("\nModel Evaluation:")
    print(f"Accuracy: {evaluation['accuracy']:.4f} ({evaluation['accuracy']*100:.2f}%)")
    print(f"Error Rate: {evaluation['error_rate']:.4f} ({evaluation['error_rate']*100:.2f}%)")
    print(f"Correct Predictions: {evaluation['correct_predictions']}/{evaluation['total_predictions']}")
    print(f"Errors: {evaluation['errors']}")

    # Save the trained model
    model_path = "models/trained_mlp"
    mlp.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    # Test loading the model
    print("\nTesting model loading...")
    loaded_mlp = MLP.load_model(model_path)

    # Test predictions with loaded model
    loaded_pred = loaded_mlp.predict(x_test)
    loaded_evaluation = loaded_mlp.evaluate(loaded_pred, y_test)

    print("Loaded Model Evaluation:")
    print(f"Accuracy: {loaded_evaluation['accuracy']:.4f} ({loaded_evaluation['accuracy']*100:.2f}%)")
    print(f"Error Rate: {loaded_evaluation['error_rate']:.4f} ({loaded_evaluation['error_rate']*100:.2f}%)")
    print(f"Correct Predictions: {loaded_evaluation['correct_predictions']}/{loaded_evaluation['total_predictions']}")
    print(f"Errors: {loaded_evaluation['errors']}")

    # Verify predictions are identical
    predictions_match = np.allclose(y_pred, loaded_pred, atol=1e-10)
    print(f"\nPredictions match between original and loaded model: {predictions_match}")


if __name__ == "__main__":
    main()
