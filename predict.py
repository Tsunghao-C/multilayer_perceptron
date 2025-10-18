#!/usr/bin/env python3
"""
Script to load a saved MLP model and make predictions on new data.
"""

import argparse

import numpy as np
import pandas as pd

from src.MLP.models import MLP
from train_test_split import columns_to_exclude


def parse_arg():
    parser = argparse.ArgumentParser(
        description="Load a trained MLP model and make predictions on new data."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to the saved model (without extension)"
    )
    parser.add_argument(
        "--data",
        required=True,
        type=str,
        help="Path to the dataset for prediction"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to save predictions (default: predictions.csv)"
    )
    return parser.parse_args()


def zscore(x):
    """Z-score normalization function."""
    return (x - np.mean(x)) / np.std(x)


def main():
    args = parse_arg()

    # Load the trained model
    print(f"Loading model from {args.model}...")
    mlp = MLP.load_model(args.model)
    print("Model loaded successfully!")

    # Load and preprocess the data
    print(f"Loading data from {args.data}...")
    data = pd.read_csv(args.data)

    # Check if this is test data with labels or new data without labels
    has_labels = "Diagnosis" in data.columns

    if has_labels:
        # This is test data with labels
        print("Test data detected (with labels)")
        # Create one-hot encoded labels
        df_one = pd.get_dummies(data["Diagnosis"], dtype=int)
        data = data.drop(["Diagnosis"], axis=1)

        # Ensure columns are in consistent order: [B, M]
        if 'B' in df_one.columns and 'M' in df_one.columns:
            y_true = df_one[['B', 'M']]
        else:
            y_true = None
    else:
        # This is new data without labels
        print("New data detected (without labels)")
        y_true = None

    # Feature selection (same as training)
    ditch_columns = ["ID", "Diagnosis", "smoothness", "symmetry", "fractal_dim"]
    selected_cols = [col for col in data.columns.tolist() if not columns_to_exclude(ditch_columns, col)]
    X = data[selected_cols]

    # Apply z-score normalization
    features = list(X.columns)
    X = X.copy()  # Avoid SettingWithCopyWarning
    for feat in features:
        X[feat] = zscore(X[feat])

    # Convert to numpy array
    X = X.values

    # Make predictions
    print("Making predictions...")
    predictions = mlp.predict(X)
    print(predictions)
    print(predictions.shape)

    # Convert probabilities to class predictions
    class_predictions = np.argmax(predictions, axis=1)
    print(class_predictions)
    print(class_predictions.shape)
    class_names = ['B', 'M']  # B=0, M=1
    predicted_classes = [class_names[i] for i in class_predictions]

    # Create results DataFrame
    results = pd.DataFrame({
        'Sample': range(len(predictions)),
        'Predicted_Class': predicted_classes,
        'P_Benign': predictions[:, 0],
        'P_Malignant': predictions[:, 1]
    })

    # Add true labels if available
    if y_true is not None:
        true_classes = []
        for i in range(len(y_true)):
            if y_true.iloc[i, 0] == 1:  # B column
                true_classes.append('B')
            else:  # M column
                true_classes.append('M')
        results['True_Class'] = true_classes
        results['Correct'] = results['Predicted_Class'] == results['True_Class']

        # Calculate accuracy
        accuracy = results['Correct'].mean()
        print(f"\nAccuracy on test data: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Save predictions
    results.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

    # Display sample results
    print("\nSample predictions:")
    print(results.head(10))


if __name__ == "__main__":
    main()
