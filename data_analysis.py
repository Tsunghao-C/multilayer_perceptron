import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_arg():
    parser = argparse.ArgumentParser(
        description="Take a dataset as input and analyse it"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        default="data/data.csv",
        help="Path to dataset"
    )
    return parser.parse_args()


def main():
    args = parse_arg()
    # read raw data
    data = pd.read_csv(str(args.dataset), header=None)
    # add headers
    headers = ["ID", "Diagnosis"]
    features_raw = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_points", "symmetry", "fractal_dim"]
    features_mean = [f"{feat}_mean" for feat in features_raw]
    features_std = [f"{feat}_std" for feat in features_raw]
    features_worst = [f"{feat}_worst" for feat in features_raw]
    headers = headers + features_mean + features_std + features_worst
    data.columns = headers
    # draw pairplots for mean, std, and worst
    for title in ["mean", "std", "worst"]:
        selected_cols = [col for col in data.columns.to_list() if col.endswith(title)]
        pairplot = sns.pairplot(
            data,
            vars=selected_cols,
            hue="Diagnosis",
            height=0.7,
            diag_kind="hist"
        )

        # Save the pairplot with appropriate filename
        filename = f"pairplot_{title}.png"
        pairplot.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved pairplot to {filename}")

        plt.close()


if __name__ == "__main__":
    main()
