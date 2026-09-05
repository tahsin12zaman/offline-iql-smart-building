import numpy as np
import pandas as pd


INPUT_METRICS = "results/citylearn_application_metrics.csv"
OUTPUT_FINAL = "results/citylearn_final_results.csv"
DATASET_PATH = "data/citylearn_logged_multi.npz"


def create_final_results():
    df = pd.read_csv(INPUT_METRICS)

    # Official controlled comparison: seeds 1, 2, and 3 only.
    df = df[df["seed"].isin([1, 2, 3])].copy()

    # Keep a consistent algorithm ordering.
    algo_order = {
        "BC": 0,
        "IQL": 1,
        "CQL": 2,
    }

    df["_order"] = df["algorithm"].map(algo_order)
    df = (
        df.sort_values(["_order", "seed"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    df.to_csv(OUTPUT_FINAL, index=False)

    print("=" * 80)
    print("FINAL CITYLEARN CONTROLLED RESULTS")
    print("=" * 80)
    print(f"Saved to: {OUTPUT_FINAL}")
    print()
    print(df.to_string(index=False))

    metrics = [
        "reward",
        "net_electricity_consumption",
        "electricity_cost",
        "carbon_emission",
        "peak_net_electricity",
    ]

    summary = df.groupby("algorithm")[metrics].agg(["mean", "std"])

    print()
    print("=" * 80)
    print("MEAN ± STD ACROSS SEEDS 1, 2, 3")
    print("=" * 80)

    for algo in ["BC", "IQL", "CQL"]:
        print(f"\n{algo}")

        for metric in metrics:
            mean = summary.loc[algo, (metric, "mean")]
            std = summary.loc[algo, (metric, "std")]

            print(
                f"  {metric:30s}: "
                f"{mean:.6f} ± {std:.6f}"
            )


def inspect_dataset_metadata():
    data = np.load(DATASET_PATH)

    print()
    print("=" * 80)
    print("CITYLEARN DATASET METADATA")
    print("=" * 80)

    fields = [
        "env_name",
        "schema_path",
        "episodes",
        "max_steps_per_episode",
        "noise_std",
        "seed",
    ]

    for key in fields:
        if key not in data.files:
            print(f"{key:25s}: NOT FOUND")
            continue

        value = data[key]

        if value.ndim == 0:
            value = value.item()

        print(f"{key:25s}: {value}")

    print()
    print("Dataset arrays:")

    for key in [
        "observations",
        "actions",
        "rewards",
        "terminals",
        "episode_ids",
    ]:
        arr = data[key]
        print(
            f"{key:25s}: shape={arr.shape}, "
            f"dtype={arr.dtype}"
        )

    episode_ids = data["episode_ids"]
    unique_ids, counts = np.unique(
        episode_ids,
        return_counts=True,
    )

    print()
    print(f"Total transitions       : {len(data['observations'])}")
    print(f"Unique trajectories     : {len(unique_ids)}")
    print(f"Terminal transitions    : {int(data['terminals'].sum())}")

    print()
    print("Transitions per trajectory:")

    for episode_id, count in zip(unique_ids, counts):
        print(f"  trajectory {episode_id:2d}: {count}")


def main():
    create_final_results()
    inspect_dataset_metadata()


if __name__ == "__main__":
    main()