import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


INPUT_CSV = "results/citylearn_application_metrics.csv"

OUTPUT_PNG = (
    "results/"
    "citylearn_per_seed_application_metrics.png"
)


def main():
    df = pd.read_csv(INPUT_CSV)

    algorithms = ["BC", "IQL", "CQL"]

    metrics = [
        (
            "net_electricity_consumption",
            "Net Electricity",
        ),
        (
            "electricity_cost",
            "Electricity Cost",
        ),
        (
            "carbon_emission",
            "Carbon Emissions",
        ),
        (
            "peak_net_electricity",
            "Peak Net Electricity",
        ),
    ]

    # Fixed seed colors so the legend matches every subplot.
    seed_colors = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
    }

    seed_offsets = {
        1: -0.10,
        2: 0.00,
        3: 0.10,
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8),
    )

    axes = axes.flatten()

    for ax, (metric, title) in zip(
        axes,
        metrics,
    ):
        for x_pos, algorithm in enumerate(
            algorithms
        ):
            algo_df = df[
                df["algorithm"] == algorithm
            ].sort_values("seed")

            values = algo_df[
                metric
            ].to_numpy()

            seeds = algo_df[
                "seed"
            ].to_numpy()

            for value, seed in zip(
                values,
                seeds,
            ):
                ax.scatter(
                    x_pos + seed_offsets[seed],
                    value,
                    s=60,
                    color=seed_colors[seed],
                    alpha=0.85,
                    label=(
                        f"Seed {seed}"
                        if x_pos == 0
                        else None
                    ),
                )

            mean_value = np.mean(values)

            ax.scatter(
                x_pos,
                mean_value,
                marker="D",
                s=95,
                color="tab:red",
                edgecolor="black",
                linewidth=1.0,
                label=(
                    "Mean"
                    if x_pos == 0
                    else None
                ),
            )

        ax.set_xticks(
            range(len(algorithms))
        )

        ax.set_xticklabels(
            algorithms
        )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            "Algorithm"
        )

        ax.set_ylabel(
            title
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.98),
    )

    fig.suptitle(
        "CityLearn Application Metrics Across Controlled Training Seeds",
        fontsize=15,
        y=1.02,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    plt.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        "Saved:",
        OUTPUT_PNG,
    )

    plt.show()


if __name__ == "__main__":
    main()