import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

from d3rlpy.datasets import get_pendulum
from d3rlpy.dataset import MDPDataset
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


def select_episodes(episodes, variant, seed):
    rng = np.random.default_rng(seed)
    episodes = list(episodes)

    if variant == "full":
        return episodes

    if variant == "half":
        n = max(1, int(len(episodes) * 0.50))
        indices = rng.choice(len(episodes), size=n, replace=False)
        return [episodes[i] for i in indices]

    if variant == "quarter":
        n = max(1, int(len(episodes) * 0.25))
        indices = rng.choice(len(episodes), size=n, replace=False)
        return [episodes[i] for i in indices]

    if variant in ["reward_noise", "action_noise"]:
        return episodes

    raise ValueError("Unknown dataset variant: " + variant)


def make_variant_dataset(original_dataset, env, variant, seed, reward_noise_scale, action_noise_scale):
    rng = np.random.default_rng(seed)

    selected_episodes = select_episodes(original_dataset.episodes, variant, seed)

    observations_list = []
    actions_list = []
    rewards_list = []
    terminals_list = []

    all_rewards = []

    for episode in selected_episodes:
        rewards = np.asarray(episode.rewards, dtype=np.float32)
        all_rewards.append(rewards.reshape(-1))

    all_rewards = np.concatenate(all_rewards)
    reward_std = float(np.std(all_rewards))
    if reward_std == 0.0:
        reward_std = 1.0

    for episode in selected_episodes:
        observations = np.asarray(episode.observations, dtype=np.float32)
        actions = np.asarray(episode.actions, dtype=np.float32)
        rewards = np.asarray(episode.rewards, dtype=np.float32)

        if variant == "reward_noise":
            noise = rng.normal(
                loc=0.0,
                scale=reward_noise_scale * reward_std,
                size=rewards.shape,
            ).astype(np.float32)
            rewards = rewards + noise

        if variant == "action_noise":
            noise = rng.normal(
                loc=0.0,
                scale=action_noise_scale,
                size=actions.shape,
            ).astype(np.float32)
            actions = actions + noise

            if hasattr(env, "action_space"):
                low = env.action_space.low
                high = env.action_space.high
                actions = np.clip(actions, low, high)

        terminals = np.zeros((len(rewards),), dtype=np.float32)
        terminals[-1] = 1.0

        observations_list.append(observations)
        actions_list.append(actions)
        rewards_list.append(rewards)
        terminals_list.append(terminals)

    observations = np.concatenate(observations_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    rewards = np.concatenate(rewards_list, axis=0)
    terminals = np.concatenate(terminals_list, axis=0)

    variant_dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    return variant_dataset, len(selected_episodes), len(observations)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", type=str, choices=["bc", "iql", "cql"], required=True)
    parser.add_argument(
        "--variant",
        type=str,
        choices=["full", "half", "quarter", "reward_noise", "action_noise"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-noise-scale", type=float, default=0.5)
    parser.add_argument("--action-noise-scale", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="results/dataset_variant_scores.csv")

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    set_seed(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else False

    print("=" * 70)
    print("Offline RL Dataset Variant Experiment")
    print("=" * 70)
    print("Algorithm:", args.algo.upper())
    print("Dataset variant:", args.variant)
    print("Training steps:", args.steps)
    print("Seed:", args.seed)
    print("Device:", device)
    print("=" * 70)

    print("Loading original offline Pendulum dataset...")
    original_dataset, env = get_pendulum()

    print("Original number of episodes:", len(original_dataset.episodes))

    dataset, num_episodes, num_transitions = make_variant_dataset(
        original_dataset=original_dataset,
        env=env,
        variant=args.variant,
        seed=args.seed,
        reward_noise_scale=args.reward_noise_scale,
        action_noise_scale=args.action_noise_scale,
    )

    print("Variant dataset created.")
    print("Variant episodes:", num_episodes)
    print("Variant transitions:", num_transitions)

    algo = build_algorithm(args.algo, device)

    evaluator = EnvironmentEvaluator(env)

    print("Starting training...")

    algo.fit(
        dataset,
        n_steps=args.steps,
        evaluators={
            "environment": evaluator,
        },
        experiment_name=(
            args.algo
            + "_"
            + args.variant
            + "_pendulum_seed_"
            + str(args.seed)
        ),
    )

    print("Training finished.")

    final_score = evaluator(algo, dataset=None)

    print("=" * 70)
    print("Final score:", final_score)
    print("=" * 70)

    row = {
        "algorithm": args.algo.upper(),
        "dataset": "pendulum",
        "variant": args.variant,
        "steps": args.steps,
        "seed": args.seed,
        "num_episodes": num_episodes,
        "num_transitions": num_transitions,
        "reward_noise_scale": args.reward_noise_scale,
        "action_noise_scale": args.action_noise_scale,
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