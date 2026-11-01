from pathlib import Path
import argparse

import d3rlpy
import numpy as np
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import IQLConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA = (
    PROJECT_ROOT
    / "data"
    / "citylearn_logged_multi.npz"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "trust_ensemble"
)


def build_iql():
    return IQLConfig(
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        expectile=0.7,
        weight_temp=3.0,
        max_weight=100.0,
    ).create(device=False)


def train_one(
    dataset,
    seed,
    steps,
    output_dir,
):
    # d3rlpy.seed sets Python random, NumPy,
    # and PyTorch random seeds.
    d3rlpy.seed(seed)

    print()
    print("=" * 78)
    print(f"TRAINING TRUST-ENSEMBLE IQL SEED {seed}")
    print("=" * 78)
    print(f"Seed:  {seed}")
    print(f"Steps: {steps}")

    algo = build_iql()

    algo.fit(
        dataset,
        n_steps=steps,
        n_steps_per_epoch=1000,
        show_progress=True,
    )

    model_path = (
        output_dir
        / f"citylearn_iql_seed_{seed}.d3"
    )

    algo.save_model(
        str(model_path)
    )

    print(f"Saved: {model_path}")

    return model_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    arr = np.load(
        data_path,
        allow_pickle=True,
    )

    observations = arr[
        "observations"
    ].astype(np.float32)

    actions = arr[
        "actions"
    ].astype(np.float32)

    rewards = arr[
        "rewards"
    ].astype(np.float32)

    terminals = arr[
        "terminals"
    ].astype(bool)

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    print("=" * 78)
    print("CITYLEARN REPRODUCIBLE IQL TRUST ENSEMBLE")
    print("=" * 78)
    print(f"Dataset:      {data_path}")
    print(f"Observations: {observations.shape}")
    print(f"Actions:      {actions.shape}")
    print(f"Steps/model:  {args.steps}")
    print(f"Seeds:        {args.seeds}")
    print(f"Output:       {output_dir}")

    paths = []

    for seed in args.seeds:
        paths.append(
            train_one(
                dataset=dataset,
                seed=seed,
                steps=args.steps,
                output_dir=output_dir,
            )
        )

    print()
    print("=" * 78)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 78)

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
