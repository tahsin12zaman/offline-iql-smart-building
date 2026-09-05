import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = "results/citylearn_net_electricity_timeseries.csv"

OUTPUT_PNG = (
    "results/"
    "citylearn_net_electricity_timeseries_smoothed.png"
)

ROLLING_WINDOW = 24


def main():
    df = pd.read_csv(INPUT_CSV)

    # Keep district-level data only.
    df = df[
        df["level"] == "district"
    ].copy()

    # Sort before applying the rolling mean.
    df = df.sort_values(
        ["algorithm", "seed", "step"]
    )

    # Apply a 24-timestep rolling mean separately
    # to each algorithm and training seed.
    df["net_electricity_smoothed"] = (
        df.groupby(
            ["algorithm", "seed"]
        )["net_electricity"]
        .transform(
            lambda x: x.rolling(
                window=ROLLING_WINDOW,
                min_periods=1,
                center=True,
            ).mean()
        )
    )

    # Compute mean and standard deviation across
    # controlled training seeds 1, 2, and 3.
    summary = (
        df.groupby(
            ["algorithm", "step"]
        )["net_electricity_smoothed"]
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
            linewidth=2.0,
            label=algorithm,
        )[0]

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.18,
            color=line.get_color(),
        )

    # Zero reference line.
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
        "District Net Electricity"
    )

    plt.title(
        "CityLearn District Net Electricity Profile Across the Evaluation Episode"
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
        OUTPUT_PNG
    )

    print(
        f"Rolling window: {ROLLING_WINDOW} timesteps"
    )

    plt.show()


if __name__ == "__main__":
    main()