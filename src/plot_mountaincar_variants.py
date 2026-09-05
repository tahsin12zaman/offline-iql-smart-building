import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = "results/mountaincar_variant_scores.csv"
OUT_PNG = "results/mountaincar_variant_comparison.png"
OUT_SUMMARY = "results/final_mountaincar_variant_summary.csv"


def main():
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    # Keep only proper 20k runs, ignore earlier 5k sanity tests
    df = df[df["steps"] == 20000].copy()

    # Remove duplicate runs if the same setting was accidentally rerun
    df = df.drop_duplicates(
        subset=["environment", "algorithm", "variant", "steps", "seed"],
        keep="last",
    )

    variant_order = ["full", "half", "quarter"]
    algorithm_order = ["BC", "IQL", "CQL"]

    df["variant"] = pd.Categorical(df["variant"], categories=variant_order, ordered=True)
    df["algorithm"] = pd.Categorical(df["algorithm"], categories=algorithm_order, ordered=True)

    summary = (
        df.groupby(["variant", "algorithm"], observed=True)["score_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["variant", "algorithm"])
    )

    summary.to_csv(OUT_SUMMARY, index=False)

    print("\nMountainCar summary:")
    print(summary)
    print("\nSaved summary to:", OUT_SUMMARY)

    x = np.arange(len(variant_order))
    width = 0.25

    plt.figure(figsize=(9, 5))

    for i, algo in enumerate(algorithm_order):
        sub = summary[summary["algorithm"] == algo]

        means = []
        stds = []

        for variant in variant_order:
            row = sub[sub["variant"] == variant]
            if len(row) == 0:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(float(row["mean"].iloc[0]))
                stds.append(float(row["std"].iloc[0]))

        positions = x + (i - 1) * width

        plt.bar(
            positions,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=algo,
        )

    plt.xticks(x, ["Full", "Half", "Quarter"])
    plt.xlabel("Offline dataset size")
    plt.ylabel("Evaluation return")
    plt.title("MountainCarContinuous-v0: Dataset-Size Robustness")
    plt.legend(title="Algorithm")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Scores are all around 94, so this zoomed axis makes small differences visible.
    # Mention this if you put the figure in the report.
    plt.ylim(90, 96)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()

    print("Saved plot to:", OUT_PNG)


if __name__ == "__main__":
    main()