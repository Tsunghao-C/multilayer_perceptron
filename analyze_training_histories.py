#!/usr/bin/env python3
"""
Script to analyze and compare training histories from different experiments.
"""

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and compare training histories from different experiments."
    )
    parser.add_argument(
        "--trainings_dir",
        type=str,
        default="trainings",
        help="Directory containing training history CSV files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="images",
        help="Directory to save visualization plots"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots instead of saving them"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="history_*",
        help="Pattern to match history files (e.g., 'history_batch_*', 'history_opt_*', or 'history_*')"
    )
    return parser.parse_args()


def load_training_histories(trainings_dir: str, pattern: str = "history_*"):
    """Load training history files matching the pattern."""
    trainings_path = Path(trainings_dir)
    if not trainings_path.exists():
        raise FileNotFoundError(f"Trainings directory not found: {trainings_dir}")

    # Find matching files
    pattern_path = trainings_path / f"{pattern}.csv"
    history_files = glob.glob(str(pattern_path))

    if not history_files:
        raise FileNotFoundError(f"No training history files found matching pattern: {pattern}")

    print(f"Found {len(history_files)} training history files:")
    for file in history_files:
        print(f"  - {Path(file).name}")

    # Load and parse histories
    histories = {}
    for file_path in history_files:
        filename = Path(file_path).name

        # Extract experiment name from filename
        # Pattern: history_batch_X_YYYYMMDD_HHMMSS.csv
        match = re.match(r'history_([^_]+)_\d{8}_\d{6}\.csv', filename)
        if match:
            exp_name = match.group(1)
        else:
            # Fallback: use filename without extension
            exp_name = filename.replace('.csv', '')

        # Load the history
        df = pd.read_csv(file_path)
        histories[exp_name] = df

        print(f"  Loaded {exp_name}: {len(df)} epochs")

    return histories


def create_training_curves_plot(histories: dict, output_dir: str, show: bool = False):
    """Create training curves comparison plot."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Colors for different experiments
    colors = plt.cm.tab10(range(len(histories)))

    # Plot 1: Training Loss
    for i, (exp_name, df) in enumerate(histories.items()):
        ax1.plot(df['epoch'], df['loss'],
                label=f'{exp_name}', color=colors[i], linewidth=2, alpha=0.8)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    for i, (exp_name, df) in enumerate(histories.items()):
        ax2.plot(df['epoch'], df['val_loss'],
                label=f'{exp_name}', color=colors[i], linewidth=2, alpha=0.8)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Training Accuracy
    for i, (exp_name, df) in enumerate(histories.items()):
        ax3.plot(df['epoch'], df['accuracy'],
                label=f'{exp_name}', color=colors[i], linewidth=2, alpha=0.8)

    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Training Accuracy')
    ax3.set_title('Training Accuracy Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.8, 1.0)

    # Plot 4: Validation Accuracy
    for i, (exp_name, df) in enumerate(histories.items()):
        ax4.plot(df['epoch'], df['val_accuracy'],
                label=f'{exp_name}', color=colors[i], linewidth=2, alpha=0.8)

    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Accuracy')
    ax4.set_title('Validation Accuracy Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.8, 1.0)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save plot
        plot_file = output_path / "training_curves_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Training curves comparison plot saved to: {plot_file}")

    plt.close()


def create_convergence_analysis(histories: dict, output_dir: str, show: bool = False):
    """Create convergence analysis plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Extract final performance metrics
    final_losses = []
    final_accuracies = []
    exp_names = []
    total_epochs = []

    for exp_name, df in histories.items():
        final_loss = df['val_loss'].iloc[-1]
        final_accuracy = df['val_accuracy'].iloc[-1]
        epochs = len(df)

        final_losses.append(final_loss)
        final_accuracies.append(final_accuracy)
        exp_names.append(exp_name)
        total_epochs.append(epochs)

    # Plot 1: Final Validation Loss vs Total Epochs
    ax1.scatter(total_epochs, final_losses, s=100, alpha=0.7, c='red')
    ax1.set_xlabel('Total Epochs')
    ax1.set_ylabel('Final Validation Loss')
    ax1.set_title('Final Validation Loss vs Training Duration')
    ax1.grid(True, alpha=0.3)

    # Add labels
    for epochs, loss, exp_name in zip(total_epochs, final_losses, exp_names, strict=False):
        ax1.annotate(f'{exp_name}\n{loss:.3f}', (epochs, loss),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Plot 2: Final Validation Accuracy vs Total Epochs
    ax2.scatter(total_epochs, final_accuracies, s=100, alpha=0.7, c='blue')
    ax2.set_xlabel('Total Epochs')
    ax2.set_ylabel('Final Validation Accuracy')
    ax2.set_title('Final Validation Accuracy vs Training Duration')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.9, 1.0)

    # Add labels
    for epochs, accuracy, exp_name in zip(total_epochs, final_accuracies, exp_names, strict=False):
        ax2.annotate(f'{exp_name}\n{accuracy:.3f}', (epochs, accuracy),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save plot
        plot_file = output_path / "convergence_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Convergence analysis plot saved to: {plot_file}")

    plt.close()


def print_training_summary(histories: dict):
    """Print a detailed training summary."""
    print("\n" + "="*70)
    print("TRAINING HISTORY ANALYSIS")
    print("="*70)

    summary_data = []

    for exp_name, df in histories.items():
        # Calculate metrics
        total_epochs = len(df)
        final_train_loss = df['loss'].iloc[-1]
        final_val_loss = df['val_loss'].iloc[-1]
        final_train_acc = df['accuracy'].iloc[-1]
        final_val_acc = df['val_accuracy'].iloc[-1]

        # Find best validation accuracy
        best_val_acc = df['val_accuracy'].max()
        best_val_acc_epoch = df['val_accuracy'].idxmax()

        # Calculate convergence speed (epochs to reach 95% of best accuracy)
        target_acc = 0.95 * best_val_acc
        convergence_epoch = None
        for i, acc in enumerate(df['val_accuracy']):
            if acc >= target_acc:
                convergence_epoch = i
                break

        summary_data.append({
            'Experiment': exp_name,
            'Total Epochs': total_epochs,
            'Final Train Loss': final_train_loss,
            'Final Val Loss': final_val_loss,
            'Final Train Acc': final_train_acc,
            'Final Val Acc': final_val_acc,
            'Best Val Acc': best_val_acc,
            'Best Val Acc Epoch': best_val_acc_epoch,
            'Convergence Epoch': convergence_epoch if convergence_epoch else "N/A"
        })

    # Create DataFrame for nice formatting
    summary_df = pd.DataFrame(summary_data)

    print("\n📊 TRAINING SUMMARY TABLE:")
    print(summary_df.to_string(index=False, float_format='%.4f'))

    # Find best performing experiments
    best_final_acc = summary_df.loc[summary_df['Final Val Acc'].idxmax()]
    fastest_convergence = summary_df[summary_df['Convergence Epoch'] != "N/A"].loc[
        summary_df[summary_df['Convergence Epoch'] != "N/A"]['Convergence Epoch'].idxmin()
    ]

    print("\n🏆 BEST FINAL VALIDATION ACCURACY:")
    print(f"   Experiment: {best_final_acc['Experiment']}")
    print(f"   Accuracy: {best_final_acc['Final Val Acc']:.4f} ({best_final_acc['Final Val Acc']*100:.2f}%)")
    print(f"   Total Epochs: {best_final_acc['Total Epochs']}")

    if fastest_convergence is not None and len(fastest_convergence) > 0:
        print("\n⚡ FASTEST CONVERGENCE:")
        print(f"   Experiment: {fastest_convergence['Experiment']}")
        print(f"   Converged at epoch: {fastest_convergence['Convergence Epoch']}")
        print(f"   Final accuracy: {fastest_convergence['Final Val Acc']:.4f}")
        print(f"   Total Epochs: {fastest_convergence['Total Epochs']}")


def main():
    args = parse_args()

    try:
        # Load training histories
        histories = load_training_histories(args.trainings_dir, args.pattern)

        # Print analysis
        print_training_summary(histories)

        # Create visualizations
        create_training_curves_plot(histories, args.output_dir, args.show)
        create_convergence_analysis(histories, args.output_dir, args.show)

        print("\n✅ Training history analysis complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
