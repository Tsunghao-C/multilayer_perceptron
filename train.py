import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display training result right after traing."
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


def plot_history(history: list[dict[str, Any]], output_file=None, show=False):
    """Plot training histories for comparison."""
    if not history:
        print("No histories to plot")
        return

    # Extract data from list of dictionaries
    epochs = [entry['epoch'] for entry in history]
    losses = [float(entry['loss']) for entry in history]  # Convert numpy types to float
    val_losses = [float(entry['val_loss']) for entry in history]  # Convert numpy types to float
    accuracies = [float(entry['accuracy']) for entry in history]  # Convert numpy types to float
    val_accuracies = [float(entry['val_accuracy']) for entry in history]  # Convert numpy types to float

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot loss - both training and validation
    ax1.plot(epochs, losses, linewidth=2, color='blue', label='Training Loss', alpha=0.8)
    ax1.plot(epochs, val_losses, linewidth=2, color='red', label='Validation Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy - both training and validation
    ax2.plot(epochs, accuracies, linewidth=2, color='blue', label='Training Accuracy', alpha=0.8)
    ax2.plot(epochs, val_accuracies, linewidth=2, color='red', label='Validation Accuracy', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training vs Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        if not output_file:
            out_dir = Path("trainings")
            out_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"train_{timestamp}.png"
            output_file = out_dir / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {output_file}")


class ZscoreScaler:
    def __init__(self) -> None:
        self.mean_ = None
        self.std_ = None
        self.feature_names_ = None

    def fit(self, X):
        """
        Learn the mean and std from training data.

        Args:
            X: pandas DataFrame or numpy array
        """
        if hasattr(X, 'values'):  # pandas DataFrame
            self.mean_ = X.mean()
            self.std_ = X.std()
            self.feature_names_ = X.columns.tolist()
        else:  # numpy array
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        """
        Apply the learned transformation to new data.

        Args:
            X: pandas DataFrame or numpy array

        Returns:
            Transformed data with same type as input
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler must be fitted before transform")

        if hasattr(X, 'values'):  # pandas DataFrame
            # Ensure same columns as training data
            if self.feature_names_ is not None:
                X = X[self.feature_names_]
            return (X - self.mean_) / self.std_
        else:  # numpy array
            return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """
        Fit the scaler and transform the data in one step.

        Args:
            X: pandas DataFrame or numpy array

        Returns:
            Transformed data with same type as input
        """
        return self.fit(X).transform(X)


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

    # 2. Apply z-score normalization using proper fit/transform approach
    # Learn transformation parameters from training data only
    scaler = ZscoreScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    # Apply same transformation to test data
    x_test_scaled = scaler.transform(x_test)

    # Convert pandas DataFrames to numpy arrays for neural network processing
    x_train = x_train_scaled.values
    y_train = y_train.values  # Keep as 2D array for one-hot encoded labels
    x_test = x_test_scaled.values
    y_test = y_test.values

    # Generate network configuration from JSON
    network_config = nn_config_gen(str(args.config), x_train.shape[1])
    # print(network_config)

    # Init MLP network instance with fixed random seed for reproducible results
    mlp = MLP(network_config, epoch=1000, lr=0.005, batch_size=32)
    print(mlp)

    # Train with input data
    mlp.fit(x_train, y_train, x_test, y_test)

    # Predict with test data
    y_pred = mlp.predict(x_test)
    # print("Sample predictions (first 5):")
    # print(y_pred[:5])
    # print("Sample ground truth (first 5):")
    # print(y_test[:5])

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

    # Enhanced metrics
    print("\nDetailed Classification Metrics:")
    class_names = ['Benign (B)', 'Malignant (M)']
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    Precision: {evaluation['precision_per_class'][i]:.4f}")
        print(f"    Recall: {evaluation['recall_per_class'][i]:.4f}")
        print(f"    F1-Score: {evaluation['f1_per_class'][i]:.4f}")

    print("\nMacro Averages:")
    print(f"  Precision: {evaluation['macro_precision']:.4f}")
    print(f"  Recall: {evaluation['macro_recall']:.4f}")
    print(f"  F1-Score: {evaluation['macro_f1']:.4f}")

    print("\nMicro Averages:")
    print(f"  Precision: {evaluation['micro_precision']:.4f}")
    print(f"  Recall: {evaluation['micro_recall']:.4f}")
    print(f"  F1-Score: {evaluation['micro_f1']:.4f}")

    # Save the trained model
    model_path = "models/trained_mlp"
    mlp.save_model(model_path)

    # Plot result
    history = mlp.history
    plot_history(history, show=args.display)

    # # Test loading the model
    # print("\nTesting model loading...")
    # loaded_mlp = MLP.load_model(model_path)

    # # Test predictions with loaded model
    # loaded_pred = loaded_mlp.predict(x_test)
    # loaded_evaluation = loaded_mlp.evaluate(loaded_pred, y_test)

    # print("Loaded Model Evaluation:")
    # print(f"Accuracy: {loaded_evaluation['accuracy']:.4f} ({loaded_evaluation['accuracy']*100:.2f}%)")
    # print(f"Error Rate: {loaded_evaluation['error_rate']:.4f} ({loaded_evaluation['error_rate']*100:.2f}%)")
    # print(f"Correct Predictions: {loaded_evaluation['correct_predictions']}/{loaded_evaluation['total_predictions']}")
    # print(f"Errors: {loaded_evaluation['errors']}")

    # # Verify predictions are identical
    # predictions_match = np.allclose(y_pred, loaded_pred, atol=1e-10)
    # print(f"\nPredictions match between original and loaded model: {predictions_match}")


if __name__ == "__main__":
    main()
