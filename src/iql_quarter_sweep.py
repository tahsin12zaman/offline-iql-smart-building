import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

from d3rlpy.datasets import get_pendulum
from d3rlpy.algos import IQLConfig
from d3rlpy.metrics import EnvironmentEvaluator

from train_dataset_variants import make_variant_dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def already_done(out_path, expectile, weight_temp, seed, steps):
    if not os.path.exists(out_path):
        return False

    df = pd.read_csv(out_path)

    matched = df[
        (df["expectile"] == expectile)
        & (df["weight_temp"] == weight_temp)
        & (df["seed"] == seed)
        & (df["steps"] == steps)
    ]

    return len(matched) > 0


def train_iql_quarter(expectile, weight_temp, seed, steps):
    set_seed(seed)

    device = "cuda:0" if torch.cuda.is_available() else False

    print("=" * 70)
    print("IQL Quarter Dataset Hyperparameter Sweep")
    print("=" * 70)
    print("Expectile:", expectile)
    print("Weight temp:", weight_temp)
    print("Seed:", seed)
    print("Steps:", steps)
    print("Device:", device)
    print("=" * 70)

    original_dataset, env = get_pendulum()

    dataset, num_episodes, num_transitions = make_variant_dataset(
        original_dataset=original_dataset,
        env=env,
        variant="quarter",
        seed=seed,
        reward_noise_scale=0.5,
        action_noise_scale=0.2,
    )

    print("Quarter dataset created.")
    print("Episodes:", num_episodes)
    print("Transitions:", num_transitions)

    algo = IQLConfig(
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        expectile=expectile,
        weight_temp=weight_temp,
        max_weight=100.0,
    ).create(device=device)

    evaluator = EnvironmentEvaluator(env)

    algo.fit(
        dataset,
        n_steps=steps,
        evaluators={
            "environment": evaluator,
        },
        experiment_name=(
            "iql_quarter"
            + "_expectile_"
            + str(expectile)
            + "_weight_"
            + str(weight_temp)
            + "_seed_"
            + str(seed)
        ),
    )

    final_score = evaluator(algo, dataset=None)

    print("=" * 70)
    print("Final score:", final_score)
    print("=" * 70)

    return {
        "algorithm": "IQL",
        "dataset": "pendulum",
        "variant": "quarter",
        "steps": steps,
        "seed": seed,
        "expectile": expectile,
        "weight_temp": weight_temp,
        "num_episodes": num_episodes,
        "num_transitions": num_transitions,
        "score": final_score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--out", type=str, default="results/iql_quarter_sweep.csv")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    seeds = [0, 1, 2]
    expectiles = [0.5, 0.7, 0.9]
    weight_temps = [1.0, 3.0, 10.0]

    for expectile in expectiles:
        for weight_temp in weight_temps:
            for seed in seeds:
                if already_done(args.out, expectile, weight_temp, seed, args.steps):
                    print(
                        "Skipping existing run:",
                        "expectile=", expectile,
                        "weight_temp=", weight_temp,
                        "seed=", seed,
                    )
                    continue

                row = train_iql_quarter(
                    expectile=expectile,
                    weight_temp=weight_temp,
                    seed=seed,
                    steps=args.steps,
                )

                if os.path.exists(args.out):
                    df = pd.read_csv(args.out)
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                else:
                    df = pd.DataFrame([row])

                df.to_csv(args.out, index=False)
                print("Saved result to:", args.out)

    print("Sweep finished.")


if __name__ == "__main__":
    main()