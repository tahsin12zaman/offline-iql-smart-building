import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


INPUT_FILE = "results/citylearn_native_kpis.csv"
OUTPUT_FILE = "results/citylearn_thermal_comfort_comparison.png"

ALGORITHMS = ["BC", "IQL", "CQL"]
LOCATIONS = ["District", "Building_1", "Building_2", "Building_3"]


def main():
    df = pd.read_csv(INPUT_FILE)

    comfort = df[
        df["cost_function"] == "discomfort_proportion"
    ].copy()

    summary = (
        comfort
        .groupby(["algorithm", "name"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    x = np.arange(len(LOCATIONS))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, algorithm in enumerate(ALGORITHMS):
        means = []
        stds = []

        for location in LOCATIONS:
            row = summary[
                (summary["algorithm"] == algorithm)
                & (summary["name"] == location)
            ]

            means.append(row.iloc[0]["mean"] * 100.0)
            stds.append(row.iloc[0]["std"] * 100.0)

        offset = (i - 1) * width

        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=algorithm,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "District",
            "Building 1",
            "Building 2",
            "Building 3",
        ]
    )

    ax.set_ylabel("Thermal Discomfort Proportion (%)")

    ax.set_title(
        "CityLearn Thermal Comfort Across the District and Buildings"
    )

    ax.legend(title="Algorithm")

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved: {OUTPUT_FILE}")

    print()
    print("Mean discomfort proportion (%)")
    print("-" * 60)

    for location in LOCATIONS:
        print(f"\n{location}")

        for algorithm in ALGORITHMS:
            row = summary[
                (summary["algorithm"] == algorithm)
                & (summary["name"] == location)
            ]

            mean = row.iloc[0]["mean"] * 100.0
            std = row.iloc[0]["std"] * 100.0

            print(
                f"  {algorithm}: "
                f"{mean:.2f} ± {std:.2f}"
            )


if __name__ == "__main__":
    main()