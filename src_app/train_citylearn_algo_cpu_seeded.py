import argparse
from pathlib import Path
import random
import numpy as np
import torch
import d3rlpy

from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import BCConfig, IQLConfig, CQLConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        d3rlpy.seed(seed)
    except Exception:
        pass


def build_algo(name):
    name = name.lower()

    if name == "bc":
        return BCConfig(
            batch_size=256,
            learning_rate=1e-3,
        ).create(device=False)

    if name == "iql":
        return IQLConfig(
            batch_size=256,
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            expectile=0.7,
            weight_temp=3.0,
            max_weight=100.0,
        ).create(device=False)

    if name == "cql":
        return CQLConfig(
            batch_size=256,
            actor_learning_rate=1e-4,
            critic_learning_rate=3e-4,
            conservative_weight=5.0,
        ).create(device=False)

    raise ValueError(f"Unknown algorithm: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["bc", "iql", "cql"], required=True)
    parser.add_argument("--data", type=str, default="data/citylearn_logged_multi.npz")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    arr = np.load(args.data, allow_pickle=True)

    observations = arr["observations"].astype(np.float32)
    actions = arr["actions"].astype(np.float32)
    rewards = arr["rewards"].astype(np.float32)
    terminals = arr["terminals"].astype(bool)

    print("=" * 70)
    print(f"CityLearn Offline RL Training: {args.algo.upper()}")
    print("=" * 70)
    print("Data:", args.data)
    print("Observations:", observations.shape)
    print("Actions:", actions.shape)
    print("Rewards:", rewards.shape)
    print("Terminals:", terminals.shape)
    print("Steps:", args.steps)
    print("Seed:", args.seed)

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    algo = build_algo(args.algo)

    algo.fit(
        dataset,
        n_steps=args.steps,
        n_steps_per_epoch=1000,
        show_progress=True,
    )

    Path("results").mkdir(exist_ok=True)
    model_path = f"results/citylearn_{args.algo}_seed_{args.seed}.d3"
    algo.save_model(model_path)

    print("=" * 70)
    print(f"{args.algo.upper()} training finished.")
    print("Saved model:", model_path)
    print("=" * 70)


if __name__ == "__main__":
    main()