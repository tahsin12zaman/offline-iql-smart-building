from pathlib import Path
import argparse
import csv
import sys

import numpy as np


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Use the CityLearn source bundled with this repository.
VENDORED_CITYLEARN = PROJECT_ROOT / "external" / "CityLearn"

if str(VENDORED_CITYLEARN) not in sys.path:
    sys.path.insert(0, str(VENDORED_CITYLEARN))


# These imports must happen after the vendored CityLearn path is added.
from citylearn.citylearn import CityLearnEnv
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import BCConfig, IQLConfig, CQLConfig


DEFAULT_SCHEMA = (
    "external/CityLearn/data/datasets/"
    "citylearn_challenge_2023_phase_2_local_evaluation/schema.json"
)


def obs_to_array(obs):
    if isinstance(obs, tuple):
        obs = obs[0]

    if isinstance(obs, list):
        obs = obs[0]

    return np.asarray(obs, dtype=np.float32).reshape(-1)


def reward_to_float(reward):
    if isinstance(reward, list):
        return float(np.sum(reward))

    return float(np.asarray(reward).sum())


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


def load_dataset(data_path):
    arr = np.load(
        data_path,
        allow_pickle=True,
    )

    observations = arr["observations"].astype(np.float32)
    actions = arr["actions"].astype(np.float32)
    rewards = arr["rewards"].astype(np.float32)
    terminals = arr["terminals"].astype(bool)

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    return dataset, observations, actions


def build_and_load_algo(
    algo_name,
    model_path,
    dataset,
    observations,
    actions,
):
    algo = build_algo(algo_name)

    try:
        algo.build_with_dataset(dataset)

    except AttributeError:
        algo.create_impl(
            observation_shape=observations.shape[1:],
            action_size=actions.shape[1],
        )

    algo.load_model(model_path)

    return algo


def evaluate_policy(
    algo,
    schema_path,
    max_steps,
):
    env = CityLearnEnv(
        str(schema_path),
        central_agent=True,
    )

    try:
        action_space = env.action_space[0]

        action_low = np.asarray(
            action_space.low,
            dtype=np.float32,
        )

        action_high = np.asarray(
            action_space.high,
            dtype=np.float32,
        )

        obs = obs_to_array(env.reset())

        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            action = algo.predict(
                obs.reshape(1, -1)
            )[0].astype(np.float32)

            action = np.clip(
                action,
                action_low,
                action_high,
            )

            step_out = env.step([action])

            if len(step_out) == 5:
                (
                    next_obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = step_out

                done = bool(
                    np.any(terminated)
                    or np.any(truncated)
                )

            elif len(step_out) == 4:
                next_obs, reward, done, info = step_out
                done = bool(np.any(done))

            else:
                raise RuntimeError(
                    "Unexpected CityLearn step output."
                )

            total_reward += reward_to_float(reward)
            steps += 1

            obs = obs_to_array(next_obs)

            if done:
                break

        return total_reward, steps

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="data/citylearn_logged_multi.npz",
    )

    parser.add_argument(
        "--schema",
        type=str,
        default=DEFAULT_SCHEMA,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=719,
    )

    parser.add_argument(
        "--out",
        type=str,
        default="results/citylearn_eval_scores.csv",
    )

    args = parser.parse_args()

    schema_path = Path(args.schema).resolve()
    out_path = Path(args.out)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset, observations, actions = load_dataset(
        args.data
    )

    rows = []

    for algo_name in ["bc", "iql", "cql"]:
        model_path = Path(
            f"results/citylearn_{algo_name}_seed_{args.seed}.d3"
        )

        print("=" * 70)
        print(f"Evaluating {algo_name.upper()}")
        print("Model:", model_path)
        print("Schema:", schema_path)

        algo = build_and_load_algo(
            algo_name=algo_name,
            model_path=str(model_path),
            dataset=dataset,
            observations=observations,
            actions=actions,
        )

        score, steps = evaluate_policy(
            algo=algo,
            schema_path=schema_path,
            max_steps=args.max_steps,
        )

        print(
            f"{algo_name.upper()} score:",
            score,
        )

        print(
            "Steps:",
            steps,
        )

        rows.append(
            {
                "algorithm": algo_name.upper(),
                "seed": args.seed,
                "score": score,
                "steps": steps,
                "max_steps": args.max_steps,
            }
        )

    file_exists = out_path.exists()

    with open(
        out_path,
        "a",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "seed",
                "score",
                "steps",
                "max_steps",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)

    print("=" * 70)
    print(
        "Saved evaluation scores to:",
        out_path,
    )
    print("=" * 70)


if __name__ == "__main__":
    main()