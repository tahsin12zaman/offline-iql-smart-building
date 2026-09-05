import numpy as np
import pandas as pd
from citylearn.citylearn import CityLearnEnv

from evaluate_citylearn_models import (
    build_and_load_algo,
    load_dataset,
    obs_to_array,
    reward_to_float,
)


SCHEMA_PATH = (
    "external/CityLearn/data/datasets/"
    "citylearn_challenge_2023_phase_2_local_evaluation/schema.json"
)

DATASET_PATH = "data/citylearn_logged_multi.npz"

MODELS = {
    "BC": {
        1: "results/citylearn_bc_seed_1.d3",
        2: "results/citylearn_bc_seed_2.d3",
        3: "results/citylearn_bc_seed_3.d3",
    },
    "IQL": {
        1: "results/citylearn_iql_seed_1.d3",
        2: "results/citylearn_iql_seed_2.d3",
        3: "results/citylearn_iql_seed_3.d3",
    },
    "CQL": {
        1: "results/citylearn_cql_seed_1.d3",
        2: "results/citylearn_cql_seed_2.d3",
        3: "results/citylearn_cql_seed_3.d3",
    },
}


def evaluate_model(
    algorithm,
    seed,
    model_path,
    dataset,
    observations,
    actions,
    max_steps=719,
):
    print()
    print("=" * 80)
    print(f"{algorithm} - seed {seed}")
    print("=" * 80)

    env = CityLearnEnv(
        schema=SCHEMA_PATH,
        central_agent=True,
    )

    algo = build_and_load_algo(
        algorithm.lower(),
        model_path,
        dataset,
        observations,
        actions,
    )

    # CityLearn central-agent action bounds.
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

        # CityLearn expects a list containing the
        # central-agent action vector.
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

    print(f"Steps: {steps}")
    print(f"Reward: {total_reward}")

    print()
    print("Native CityLearn evaluation:")

    evaluation = env.evaluate()

    print(evaluation)

    return evaluation


def main():

    print("=" * 80)
    print("LOADING OFFLINE DATASET")
    print("=" * 80)

    dataset, observations, actions = load_dataset(
        DATASET_PATH
    )

    print(
        "Observations shape:",
        observations.shape,
    )

    print(
        "Actions shape:",
        actions.shape,
    )

    all_results = []

    for algorithm, seeds in MODELS.items():

        for seed, model_path in seeds.items():

            evaluation = evaluate_model(
                algorithm,
                seed,
                model_path,
                dataset,
                observations,
                actions,
            )

            if isinstance(evaluation, pd.DataFrame):

                temp = evaluation.copy()

                temp.insert(
                    0,
                    "seed",
                    seed,
                )

                temp.insert(
                    0,
                    "algorithm",
                    algorithm,
                )

                all_results.append(temp)

            else:
                print()
                print(
                    "WARNING: env.evaluate() did not return "
                    "a pandas DataFrame."
                )

                print(
                    "Returned type:",
                    type(evaluation),
                )

    if all_results:

        final = pd.concat(
            all_results,
            ignore_index=True,
        )

        output_path = (
            "results/citylearn_native_kpis.csv"
        )

        final.to_csv(
            output_path,
            index=False,
        )

        print()
        print("=" * 80)
        print("SAVED NATIVE CITYLEARN KPIS")
        print("=" * 80)

        print(output_path)
        print()
        print(final.to_string(index=False))


if __name__ == "__main__":
    main()