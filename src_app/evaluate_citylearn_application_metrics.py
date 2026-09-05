from pathlib import Path
import argparse
import csv
import numpy as np

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

    raise ValueError(name)


def load_training_dataset(path):
    arr = np.load(path, allow_pickle=True)

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


def load_model(algo_name, seed, dataset, observations, actions):
    algo = build_algo(algo_name)

    try:
        algo.build_with_dataset(dataset)
    except AttributeError:
        algo.create_impl(
            observation_shape=observations.shape[1:],
            action_size=actions.shape[1],
        )

    model_path = f"results/citylearn_{algo_name}_seed_{seed}.d3"
    algo.load_model(model_path)

    return algo


def sum_building_metric(env, attribute_name):
    total = 0.0

    for building in env.buildings:
        values = getattr(building, attribute_name)

        arr = np.asarray(values, dtype=np.float64)

        total += float(np.nansum(arr))

    return total


def max_building_metric(env, attribute_name):
    values = []

    for building in env.buildings:
        arr = np.asarray(
            getattr(building, attribute_name),
            dtype=np.float64,
        )

        values.extend(arr.reshape(-1).tolist())

    return float(np.nanmax(values))


def evaluate(algo, schema, max_steps):
    env = CityLearnEnv(
        str(schema),
        central_agent=True,
    )

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
            next_obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            next_obs, reward, done, info = step_out
            done = bool(done)

        total_reward += reward_to_float(reward)
        steps += 1

        obs = obs_to_array(next_obs)

        if done:
            break

    metrics = {
        "reward": total_reward,
        "steps": steps,

        "net_electricity_consumption":
            sum_building_metric(
                env,
                "net_electricity_consumption",
            ),

        "electricity_cost":
            sum_building_metric(
                env,
                "net_electricity_consumption_cost",
            ),

        "carbon_emission":
            sum_building_metric(
                env,
                "net_electricity_consumption_emission",
            ),

        "peak_net_electricity":
            max_building_metric(
                env,
                "net_electricity_consumption",
            ),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/citylearn_logged_multi.npz",
    )

    parser.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=719,
    )

    parser.add_argument(
        "--out",
        default="results/citylearn_application_metrics.csv",
    )

    args = parser.parse_args()

    schema = Path(args.schema).resolve()

    dataset, observations, actions = load_training_dataset(
        args.data
    )

    rows = []

    for seed in args.seeds:
        for algo_name in ["bc", "iql", "cql"]:

            print("=" * 70)
            print(
                f"Evaluating {algo_name.upper()} "
                f"seed {seed}"
            )

            algo = load_model(
                algo_name,
                seed,
                dataset,
                observations,
                actions,
            )

            metrics = evaluate(
                algo,
                schema,
                args.max_steps,
            )

            row = {
                "algorithm": algo_name.upper(),
                "seed": seed,
                **metrics,
            }

            rows.append(row)

            print(
                "Reward:",
                metrics["reward"]
            )

            print(
                "Net electricity:",
                metrics["net_electricity_consumption"]
            )

            print(
                "Electricity cost:",
                metrics["electricity_cost"]
            )

            print(
                "Carbon emission:",
                metrics["carbon_emission"]
            )

            print(
                "Peak net electricity:",
                metrics["peak_net_electricity"]
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)

    fieldnames = [
        "algorithm",
        "seed",
        "reward",
        "steps",
        "net_electricity_consumption",
        "electricity_cost",
        "carbon_emission",
        "peak_net_electricity",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=" * 70)
    print("Saved:", out_path)
    print("=" * 70)


if __name__ == "__main__":
    main()