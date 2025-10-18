#!/usr/bin/env python3
"""
Script to analyze and compare training histories from different model runs.
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze and compare training histories from different model runs."
    )
    parser.add_argument(
        "--trainings_dir",
        type=str,
        default="trainings",
        help="Directory containing training history CSV files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_analysis.png",
        help="Output file for the analysis plot"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot instead of saving it"
    )
    return parser.parse_args()


def load_history_files(trainings_dir):
    """Load all history files from the trainings directory."""
    history_files = glob.glob(os.path.join(trainings_dir, "history_*.csv"))

    if not history_files:
        print(f"No history files found in {trainings_dir}")
        return []
    histories = []
    for file_path in history_files:
        try:
            df = pd.read_csv(file_path)
            # Extract timestamp from filename
            filename = Path(file_path).stem
            timestamp = filename.replace("history_", "")
            df['timestamp'] = timestamp
            df['file'] = file_path
            histories.append(df)
            print(f"Loaded {file_path}: {len(df)} epochs")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return histories


def analyze_histories(histories):
    """Analyze training histories and print summary statistics."""
    if not histories:
        print("No histories to analyze")
        return

    print("\n" + "="*60)
    print("TRAINING HISTORY ANALYSIS")
    print("="*60)

    for i, history in enumerate(histories):
        print(f"\nModel {i+1} ({history['timestamp'].iloc[0]}):")
        print(f"  Total epochs: {len(history)}")
        print(f"  Final loss: {history['loss'].iloc[-1]:.6f}")
        print(f"  Final accuracy: {history['accuracy'].iloc[-1]:.4f}")
        print(f"  Best accuracy: {history['accuracy'].max():.4f}")
        print(f"  Loss improvement: {history['loss'].iloc[0] - history['loss'].iloc[-1]:.6f}")


def plot_histories(histories, output_file=None, show=False):
    """Plot training histories for comparison."""
    if not histories:
        print("No histories to plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for i, history in enumerate(histories):
        color = colors[i % len(colors)]
        label = f"Model {i+1} ({history['timestamp'].iloc[0]})"

        # Plot loss
        ax1.plot(history['epoch'], history['loss'],
                color=color, label=label, linewidth=2)

        # Plot accuracy
        ax2.plot(history['epoch'], history['accuracy'],
                color=color, label=label, linewidth=2)

    # Configure loss plot
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Configure accuracy plot
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Analysis plot saved to {output_file}")


def main():
    """Main function."""
    args = parse_args()

    print(f"Loading training histories from {args.trainings_dir}...")
    histories = load_history_files(args.trainings_dir)

    if not histories:
        return

    # Analyze histories
    analyze_histories(histories)

    # Plot histories
    plot_histories(histories, args.output, args.show)


if __name__ == "__main__":
    main()
