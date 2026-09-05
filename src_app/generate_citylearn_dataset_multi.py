from pathlib import Path
import argparse
import numpy as np
from citylearn.citylearn import CityLearnEnv


DEFAULT_SCHEMA = (
    "external/CityLearn/data/datasets/"
    "citylearn_challenge_2023_phase_2_local_evaluation/schema.json"
)


def unwrap_reset(reset_out):
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


def obs_to_array(obs):
    if isinstance(obs, list):
        obs = obs[0]
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def reward_to_float(reward):
    if isinstance(reward, list):
        return float(np.sum(reward))
    return float(np.asarray(reward).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=str, default=DEFAULT_SCHEMA)
    parser.add_argument("--out", type=str, default="data/citylearn_logged_multi.npz")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps-per-episode", type=int, default=720)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.25)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    schema_path = Path(args.schema).resolve()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Generating Multi-Episode CityLearn Offline Dataset")
    print("=" * 70)
    print("Schema:", schema_path)
    print("Output:", out_path)
    print("Episodes:", args.episodes)
    print("Max steps per episode:", args.max_steps_per_episode)
    print("Seed:", args.seed)
    print("Noise std:", args.noise_std)

    env = CityLearnEnv(str(schema_path), central_agent=True)

    action_space = env.action_space[0]
    action_low = np.asarray(action_space.low, dtype=np.float32)
    action_high = np.asarray(action_space.high, dtype=np.float32)
    action_dim = action_space.shape[0]

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Action dimension:", action_dim)

    observations = []
    actions = []
    rewards = []
    terminals = []
    episode_ids = []

    total_reward = 0.0

    for episode in range(args.episodes):
        reset_out = env.reset()
        obs = obs_to_array(unwrap_reset(reset_out))

        episode_reward = 0.0
        episode_steps = 0

        for step in range(args.max_steps_per_episode):
            base_action = np.zeros(action_dim, dtype=np.float32)
            noise = rng.normal(0.0, args.noise_std, size=action_dim).astype(np.float32)
            action = np.clip(base_action + noise, action_low, action_high).astype(np.float32)

            step_out = env.step([action])

            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, info = step_out
                done = bool(done)

            reward_value = reward_to_float(reward)
            is_last_step = step == args.max_steps_per_episode - 1
            terminal = done or is_last_step

            observations.append(obs)
            actions.append(action)
            rewards.append([reward_value])
            terminals.append(terminal)
            episode_ids.append(episode)

            episode_reward += reward_value
            total_reward += reward_value
            episode_steps += 1

            obs = obs_to_array(next_obs)

            if done:
                break

        print(
            f"Episode {episode + 1:03d}/{args.episodes} | "
            f"steps={episode_steps} | reward={episode_reward:.2f}"
        )

    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    terminals = np.asarray(terminals, dtype=bool)
    episode_ids = np.asarray(episode_ids, dtype=np.int32)

    np.savez_compressed(
        out_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        episode_ids=episode_ids,
        action_low=action_low,
        action_high=action_high,
        env_name="CityLearn-HVAC-Energy-Control",
        schema_path=str(schema_path),
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    print("=" * 70)
    print("Dataset saved.")
    print("Transitions:", len(observations))
    print("Observation shape:", observations.shape)
    print("Action shape:", actions.shape)
    print("Reward shape:", rewards.shape)
    print("Terminal shape:", terminals.shape)
    print("Episodes:", len(np.unique(episode_ids)))
    print("Total logged reward:", total_reward)
    print("Average reward per step:", total_reward / max(1, len(observations)))
    print("=" * 70)


if __name__ == "__main__":
    main()