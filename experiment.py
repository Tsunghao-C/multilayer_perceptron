import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.MLP.models import MLP
from train import nn_config_gen
from train_test_split import columns_to_exclude, train_test_split


@dataclass
class ExperimentConfig:
    """
    Data model for configs of an MLP instance
    """
    exp_name: str
    epoch: int = 1000
    lr: float = 0.01
    es_threshold: float = 0.0001
    batch_size: int = 8
    optimizer: str = "sgd"
    # Optimizer-specific parameters
    beta1: float = None  # Adam: exponential decay rate for first moment estimates
    beta2: float = None  # Adam: exponential decay rate for second moment estimates
    rho: float = None    # RMSprop: decay rate for moving average of squared gradients
    epsilon: float = None  # Adam/RMSprop: small constant for numerical stability


def parse_arg():
    parser = argparse.ArgumentParser(
        description="Take path to training dataset and experiment configs for MLP experiments."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        default="data/data_with_headers.csv",
        help="Path to training dataset"
    )
    parser.add_argument(
        "--nn_config",
        type=str,
        default="configs/nn/nn_config.json",
        help="path to neural network configuration"
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="configs/mlp/exp_1_batch_size.json",
        help="Display training result right after traing."
    )
    return parser.parse_args()


def exp_config_gen(config_path: str) -> list[ExperimentConfig]:
    output_confs = []

    with open(config_path) as f:
        exp_confs = json.load(f)
        # exp_confs is dict["exp_label", dict[attribute, value]]

    for exp in exp_confs.keys():
        exp_name = exp
        exp_cfg = exp_confs[exp]
        cfg = ExperimentConfig(
            exp_name=exp_name,
            epoch=exp_cfg["epoch"],
            lr=exp_cfg["lr"],
            es_threshold=exp_cfg["es_threshold"],
            batch_size=exp_cfg["batch_size"],
            optimizer=exp_cfg["optimizer"],
            # Optional optimizer-specific parameters
            beta1=exp_cfg.get("beta1"),
            beta2=exp_cfg.get("beta2"),
            rho=exp_cfg.get("rho"),
            epsilon=exp_cfg.get("epsilon")
        )
        output_confs.append(cfg)

    return output_confs


def main():
    args = parse_arg()
    data = pd.read_csv(str(args.dataset))

    # Generate one-hot encoded result of B and M
    df_one = pd.get_dummies(data["Diagnosis"], dtype=int)
    data = data.drop(["Diagnosis"], axis=1)

    # feature selection
    ditch_columns = ["ID", "Diagnosis", "smoothness", "symmetry", "fractal_dim"]
    selected_cols = [col for col in data.columns.tolist() if not columns_to_exclude(ditch_columns, col)]

    X = data[selected_cols]
    y = df_one

    # Ensure columns are in consistent order: [B, M]
    if 'B' in y.columns and 'M' in y.columns:
        y = y[['B', 'M']]  # B=0, M=1 becomes [1,0] for B, [0,1] for M

    # train test split with fixed random seed for reproducible results
    x_train, x_test, y_train, y_test = train_test_split(X, y, 0.2, True, random_state=42)

    # read experiment configs
    network_config = nn_config_gen(str(args.nn_config), x_train.shape[1])
    exp_configs = exp_config_gen(str(args.exp_config))


    # Iterate through experiment configs to run MLP training with different configs
    exp_names = []
    losses = []
    accuracies = []
    for exp in exp_configs:
        print("=" * 42)
        print(f"Running experiment {exp.exp_name}")
        print("=" * 42)
        # Prepare optimizer-specific parameters
        optimizer_kwargs = {}
        if exp.beta1 is not None:
            optimizer_kwargs['beta1'] = exp.beta1
        if exp.beta2 is not None:
            optimizer_kwargs['beta2'] = exp.beta2
        if exp.rho is not None:
            optimizer_kwargs['rho'] = exp.rho
        if exp.epsilon is not None:
            optimizer_kwargs['epsilon'] = exp.epsilon

        mlp = MLP(
            network_config=network_config,
            epoch=exp.epoch,
            lr=exp.lr,
            batch_size=exp.batch_size,
            es_threshold=exp.es_threshold,
            optimizer=exp.optimizer,
            label=exp.exp_name,
            **optimizer_kwargs
        )
        x_train_scaled, _, x_test_scaled = mlp.preprocess_data(x_train, x_test=x_test)
        x_train_exp = x_train_scaled.values
        y_train_exp = y_train.values  # Keep as 2D array for one-hot encoded labels
        x_test_exp = x_test_scaled.values
        y_test_exp = y_test.values

        mlp.fit(x_train_exp, y_train_exp, x_test_exp, y_test_exp)

        y_pred_exp = mlp.predict(x_test_exp)
        loss_exp = mlp.loss_function.loss(y_test_exp, y_pred_exp, eps=1e-15)
        evluation_exp = mlp.evaluate(y_pred_exp, y_test_exp)

        # update key metrics of this experiment to list
        exp_names.append(exp.exp_name)
        losses.append(loss_exp)
        accuracies.append(evluation_exp['accuracy'])

        # save this experiment model
        model_path = f"models/trained_mlp_{exp.exp_name}"
        mlp.save_model(model_path)

    # Save experiment results to CSV
    results_df = pd.DataFrame({
        'experiment_name': exp_names,
        'loss': losses,
        'accuracy': accuracies
    })

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/experiment_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)

    print("\n" + "=" * 50)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 50)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {results_file}")

    # Find best performing experiment
    best_idx = results_df['accuracy'].idxmax()
    best_exp = results_df.iloc[best_idx]
    print(f"\n🏆 BEST EXPERIMENT: {best_exp['experiment_name']}")
    print(f"   Accuracy: {best_exp['accuracy']:.4f} ({best_exp['accuracy']*100:.2f}%)")
    print(f"   Loss: {best_exp['loss']:.4f}")


if __name__ == "__main__":
    main()
