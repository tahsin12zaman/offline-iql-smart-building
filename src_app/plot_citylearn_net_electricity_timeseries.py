import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = "results/citylearn_net_electricity_timeseries.csv"

OUTPUT_PNG = (
    "results/"
    "citylearn_net_electricity_timeseries_comparison.png"
)


def main():
    df = pd.read_csv(INPUT_CSV)

    # Keep only district-level rows.
    df = df[
        df["level"] == "district"
    ].copy()

    # Mean and standard deviation across seeds 1, 2, 3
    # for each algorithm at every timestep.
    summary = (
        df.groupby(
            ["algorithm", "step"]
        )["net_electricity"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(
        figsize=(12, 6)
    )

    for algorithm in ["BC", "IQL", "CQL"]:
        algo_df = summary[
            summary["algorithm"] == algorithm
        ]

        x = algo_df["step"].to_numpy()
        mean = algo_df["mean"].to_numpy()
        std = algo_df["std"].fillna(0).to_numpy()

        line = plt.plot(
            x,
            mean,
            linewidth=1.7,
            label=algorithm,
        )[0]

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.18,
            color=line.get_color(),
        )

    plt.axhline(
        y=0,
        linewidth=0.8,
        linestyle="--",
        color="black",
        alpha=0.5,
    )

    plt.xlabel(
        "CityLearn Evaluation Timestep"
    )

    plt.ylabel(
        "District Net Electricity Consumption"
    )

    plt.title(
        "CityLearn District Net Electricity Over Time"
    )

    plt.legend(
        title="Algorithm"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

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