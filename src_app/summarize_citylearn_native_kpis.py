import pandas as pd


INPUT_FILE = "results/citylearn_native_kpis.csv"

OUTPUT_DISTRICT = (
    "results/citylearn_native_district_kpi_summary.csv"
)

OUTPUT_COMFORT = (
    "results/citylearn_comfort_summary.csv"
)


ALGORITHM_ORDER = ["BC", "IQL", "CQL"]


def main():
    df = pd.read_csv(INPUT_FILE)

    print("=" * 80)
    print("CITYLEARN NATIVE KPI SUMMARY")
    print("=" * 80)

    print("\nRows:", len(df))
    print("Algorithms:", df["algorithm"].unique())
    print("Seeds:", sorted(df["seed"].unique()))

    # ---------------------------------------------------------
    # 1. District-level KPI summary
    # ---------------------------------------------------------

    district = df[df["level"] == "district"].copy()

    district_summary = (
        district
        .groupby(
            ["algorithm", "cost_function"]
        )["value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    district_summary.to_csv(
        OUTPUT_DISTRICT,
        index=False,
    )

    print()
    print("=" * 80)
    print("DISTRICT KPI MEAN ± STD")
    print("=" * 80)

    important_district_kpis = [
        "electricity_consumption_total",
        "cost_total",
        "carbon_emissions_total",
        "all_time_peak_average",
        "daily_peak_average",
        "ramping_average",
        "discomfort_proportion",
        "discomfort_hot_proportion",
        "discomfort_cold_proportion",
    ]

    for metric in important_district_kpis:

        print(f"\n{metric}")

        subset = district_summary[
            district_summary["cost_function"] == metric
        ]

        for algo in ALGORITHM_ORDER:

            row = subset[
                subset["algorithm"] == algo
            ]

            if len(row) == 0:
                continue

            mean = row.iloc[0]["mean"]
            std = row.iloc[0]["std"]

            print(
                f"  {algo:4s}: "
                f"{mean:.6f} ± {std:.6f}"
            )

    # ---------------------------------------------------------
    # 2. Building-level comfort analysis
    # ---------------------------------------------------------

    comfort_metrics = [
        "discomfort_proportion",
        "discomfort_hot_proportion",
        "discomfort_cold_proportion",
        "discomfort_hot_delta_average",
        "discomfort_cold_delta_average",
    ]

    comfort = df[
        (df["level"] == "building")
        & (df["cost_function"].isin(comfort_metrics))
    ].copy()

    comfort_summary = (
        comfort
        .groupby(
            [
                "algorithm",
                "name",
                "cost_function",
            ]
        )["value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    comfort_summary.to_csv(
        OUTPUT_COMFORT,
        index=False,
    )

    print()
    print("=" * 80)
    print("BUILDING-LEVEL COMFORT")
    print("=" * 80)

    for building in sorted(
        comfort["name"].unique()
    ):

        print()
        print("-" * 80)
        print(building)
        print("-" * 80)

        for metric in comfort_metrics:

            print(f"\n{metric}")

            subset = comfort_summary[
                (comfort_summary["name"] == building)
                & (
                    comfort_summary["cost_function"]
                    == metric
                )
            ]

            for algo in ALGORITHM_ORDER:

                row = subset[
                    subset["algorithm"] == algo
                ]

                if len(row) == 0:
                    continue

                mean = row.iloc[0]["mean"]
                std = row.iloc[0]["std"]

                print(
                    f"  {algo:4s}: "
                    f"{mean:.6f} ± {std:.6f}"
                )

    print()
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(OUTPUT_DISTRICT)
    print(OUTPUT_COMFORT)


if __name__ == "__main__":
    main()