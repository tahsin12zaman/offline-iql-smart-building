import argparse
import os
import random

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


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


def behavior_policy(obs, env, rng, expert_prob, noise_std):
    low = env.action_space.low.astype(np.float32)
    high = env.action_space.high.astype(np.float32)

    if rng.random() < expert_prob:
        velocity = float(obs[1])
        base_action = 1.0 if velocity >= 0 else -1.0
        action = np.array([base_action], dtype=np.float32)
        noise = rng.normal(0.0, noise_std, size=action.shape).astype(np.float32)
        action = action + noise
    else:
        action = rng.uniform(low, high).astype(np.float32)

    action = np.clip(action, low, high)
    return action.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expert-prob", type=float, default=0.8)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", type=str, default="data/mountaincar_logged.npz")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rng = np.random.default_rng(args.seed)
    env = gym.make(args.env)

    if args.max_steps is not None:
        max_steps = args.max_steps
    elif hasattr(env, "spec") and env.spec is not None and env.spec.max_episode_steps is not None:
        max_steps = env.spec.max_episode_steps
    else:
        max_steps = 999

    observations = []
    actions = []
    rewards = []
    terminals = []
    episode_ids = []
    episode_returns = []
    episode_lengths = []

    print("=" * 70)
    print("Generating offline dataset")
    print("=" * 70)
    print("Environment:", args.env)
    print("Episodes:", args.episodes)
    print("Max steps per episode:", max_steps)
    print("Expert probability:", args.expert_prob)
    print("Action noise std:", args.noise_std)
    print("=" * 70)

    for episode in range(args.episodes):
        obs = reset_env(env, args.seed + episode)

        total_reward = 0.0
        length = 0

        for t in range(max_steps):
            action = behavior_policy(
                obs=obs,
                env=env,
                rng=rng,
                expert_prob=args.expert_prob,
                noise_std=args.noise_std,
            )

            next_obs, reward, done, info = step_env(env, action)

            observations.append(np.asarray(obs, dtype=np.float32))
            actions.append(np.asarray(action, dtype=np.float32))
            rewards.append([reward])
            terminals.append(1.0 if done else 0.0)
            episode_ids.append(episode)

            total_reward += reward
            length += 1
            obs = next_obs

            if done:
                break

        episode_returns.append(total_reward)
        episode_lengths.append(length)

        if (episode + 1) % 10 == 0:
            print(
                "Collected episode",
                episode + 1,
                "/",
                args.episodes,
                "| return:",
                round(total_reward, 2),
                "| length:",
                length,
            )

    env.close()

    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    terminals = np.asarray(terminals, dtype=np.float32)
    episode_ids = np.asarray(episode_ids, dtype=np.int32)

    np.savez(
        args.out,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        episode_ids=episode_ids,
        action_low=env.action_space.low.astype(np.float32),
        action_high=env.action_space.high.astype(np.float32),
        env_name=args.env,
    )

    print("=" * 70)
    print("Dataset saved to:", args.out)
    print("Total transitions:", len(observations))
    print("Observation shape:", observations.shape)
    print("Action shape:", actions.shape)
    print("Reward shape:", rewards.shape)
    print("Terminal shape:", terminals.shape)
    print("Average episode return:", float(np.mean(episode_returns)))
    print("Average episode length:", float(np.mean(episode_lengths)))
    print("=" * 70)


if __name__ == "__main__":
    main()