import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

from d3rlpy.datasets import get_pendulum
from d3rlpy.algos import BCConfig, IQLConfig, CQLConfig
from d3rlpy.metrics import EnvironmentEvaluator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_algorithm(algo_name, device):
    algo_name = algo_name.lower()

    if algo_name == "bc":
        return BCConfig(
            batch_size=256,
            learning_rate=1e-3,
        ).create(device=device)

    if algo_name == "iql":
        return IQLConfig(
            batch_size=256,
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            expectile=0.7,
            weight_temp=3.0,
            max_weight=100.0,
        ).create(device=device)

    if algo_name == "cql":
        return CQLConfig(
            batch_size=256,
            actor_learning_rate=1e-4,
            critic_learning_rate=3e-4,
            conservative_weight=5.0,
        ).create(device=device)

    raise ValueError("Unknown algorithm: " + algo_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["bc", "iql", "cql"], required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/scores.csv")

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = False

    print("=" * 60)
    print("Offline RL Baseline Training")
    print("=" * 60)
    print("Algorithm:", args.algo.upper())
    print("Training steps:", args.steps)
    print("Seed:", args.seed)
    print("Device:", device)
    print("=" * 60)

    print("Loading offline Pendulum dataset...")
    dataset, env = get_pendulum()

    print("Dataset loaded.")
    print("Number of episodes:", len(dataset.episodes))

    algo = build_algorithm(args.algo, device)

    evaluator = EnvironmentEvaluator(env)

    print("Starting training...")

    algo.fit(
        dataset,
        n_steps=args.steps,
        evaluators={
            "environment": evaluator,
        },
        experiment_name=args.algo + "_pendulum_seed_" + str(args.seed),
    )

    print("Training finished.")

    final_score = evaluator(algo, dataset=None)

    print("=" * 60)
    print("Final score for", args.algo.upper(), ":", final_score)
    print("=" * 60)

    row = {
        "algorithm": args.algo.upper(),
        "dataset": "pendulum",
        "steps": args.steps,
        "seed": args.seed,
        "score": final_score,
    }

    if os.path.exists(args.out):
        df = pd.read_csv(args.out)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(args.out, index=False)

    print("Saved result to:", args.out)


if __name__ == "__main__":
    main()