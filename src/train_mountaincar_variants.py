import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

try:
    import gymnasium as gym
except ImportError:
    import gym

from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import BCConfig, IQLConfig, CQLConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def reset_env(env, seed):
    try:
        output = env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
        output = env.reset()

    if isinstance(output, tuple):
        return output[0]
    return output


def step_env(env, action):
    output = env.step(action)

    if len(output) == 5:
        next_obs, reward, terminated, truncated, info = output
        done = terminated or truncated
    else:
        next_obs, reward, done, info = output

    return next_obs, float(reward), bool(done), info


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


def make_variant_dataset(data_path, variant, seed, reward_noise_scale, action_noise_scale):
    rng = np.random.default_rng(seed)

    data = np.load(data_path)

    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    rewards = data["rewards"].astype(np.float32)
    terminals = data["terminals"].astype(np.float32)
    episode_ids = data["episode_ids"].astype(np.int32)

    action_low = data["action_low"].astype(np.float32)
    action_high = data["action_high"].astype(np.float32)

    unique_episodes = np.unique(episode_ids)

    if variant == "full":
        selected_episodes = unique_episodes

    elif variant == "half":
        n = max(1, int(len(unique_episodes) * 0.50))
        selected_episodes = rng.choice(unique_episodes, size=n, replace=False)

    elif variant == "quarter":
        n = max(1, int(len(unique_episodes) * 0.25))
        selected_episodes = rng.choice(unique_episodes, size=n, replace=False)

    elif variant in ["reward_noise", "action_noise"]:
        selected_episodes = unique_episodes

    else:
        raise ValueError("Unknown variant: " + variant)

    mask = np.isin(episode_ids, selected_episodes)

    observations = observations[mask]
    actions = actions[mask]
    rewards = rewards[mask]
    terminals = terminals[mask]
    episode_ids = episode_ids[mask]

    if variant == "reward_noise":
        reward_std = float(np.std(rewards))
        if reward_std == 0.0:
            reward_std = 1.0

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
        actions = np.clip(actions, action_low, action_high)

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    return dataset, len(np.unique(episode_ids)), len(observations)


def evaluate_policy(algo, env_name, seed, n_episodes):
    env = gym.make(env_name)

    if hasattr(env, "spec") and env.spec is not None and env.spec.max_episode_steps is not None:
        max_steps = env.spec.max_episode_steps
    else:
        max_steps = 999

    returns = []

    for episode in range(n_episodes):
        obs = reset_env(env, seed + 10000 + episode)

        total_reward = 0.0

        for t in range(max_steps):
            obs_batch = np.asarray([obs], dtype=np.float32)
            action = algo.predict(obs_batch)[0]
            action = np.asarray(action, dtype=np.float32)

            obs, reward, done, info = step_env(env, action)

            total_reward += reward

            if done:
                break

        returns.append(total_reward)

    env.close()

    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", type=str, choices=["bc", "iql", "cql"], required=True)
    parser.add_argument(
        "--variant",
        type=str,
        choices=["full", "half", "quarter", "reward_noise", "action_noise"],
        required=True,
    )
    parser.add_argument("--data", type=str, default="data/mountaincar_logged.npz")
    parser.add_argument("--env", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--reward-noise-scale", type=float, default=0.5)
    parser.add_argument("--action-noise-scale", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="results/mountaincar_variant_scores.csv")

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            "Dataset file not found. Generate it first with: "
            "python3 src/generate_mountaincar_dataset.py --episodes 120 --seed 0"
        )

    set_seed(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else False

    print("=" * 70)
    print("Environment 2 Offline RL Experiment")
    print("=" * 70)
    print("Environment:", args.env)
    print("Algorithm:", args.algo.upper())
    print("Variant:", args.variant)
    print("Steps:", args.steps)
    print("Seed:", args.seed)
    print("Device:", device)
    print("=" * 70)

    dataset, num_episodes, num_transitions = make_variant_dataset(
        data_path=args.data,
        variant=args.variant,
        seed=args.seed,
        reward_noise_scale=args.reward_noise_scale,
        action_noise_scale=args.action_noise_scale,
    )

    print("Variant dataset created.")
    print("Episodes:", num_episodes)
    print("Transitions:", num_transitions)

    algo = build_algorithm(args.algo, device)

    print("Starting training...")

    algo.fit(
        dataset,
        n_steps=args.steps,
        n_steps_per_epoch=1000,
        experiment_name=(
            args.algo
            + "_mountaincar_"
            + args.variant
            + "_seed_"
            + str(args.seed)
        ),
    )

    print("Training finished.")

    eval_mean, eval_std = evaluate_policy(
        algo=algo,
        env_name=args.env,
        seed=args.seed,
        n_episodes=args.eval_episodes,
    )

    print("=" * 70)
    print("Evaluation mean:", eval_mean)
    print("Evaluation std:", eval_std)
    print("=" * 70)

    row = {
        "environment": args.env,
        "algorithm": args.algo.upper(),
        "variant": args.variant,
        "steps": args.steps,
        "seed": args.seed,
        "num_episodes": num_episodes,
        "num_transitions": num_transitions,
        "eval_episodes": args.eval_episodes,
        "reward_noise_scale": args.reward_noise_scale,
        "action_noise_scale": args.action_noise_scale,
        "score_mean": eval_mean,
        "score_std": eval_std,
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