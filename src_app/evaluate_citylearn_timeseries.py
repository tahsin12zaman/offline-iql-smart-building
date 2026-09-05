from pathlib import Path

import numpy as np
import pandas as pd

from citylearn.citylearn import CityLearnEnv

from evaluate_citylearn_models import (
    load_dataset,
    build_and_load_algo,
    obs_to_array,
)


DATA_PATH = "data/citylearn_logged_multi.npz"

SCHEMA_PATH = Path(
    "external/CityLearn/data/datasets/"
    "citylearn_challenge_2023_phase_2_local_evaluation/schema.json"
)

ALGORITHMS = ["bc", "iql", "cql"]
SEEDS = [1, 2, 3]
MAX_STEPS = 719

OUTPUT_PATH = Path(
    "results/citylearn_net_electricity_timeseries.csv"
)


def evaluate_timeseries(
    algo_name,
    seed,
    dataset,
    observations,
    actions,
):
    model_path = Path(
        f"results/citylearn_{algo_name}_seed_{seed}.d3"
    )

    print("=" * 70)
    print(f"Evaluating {algo_name.upper()} seed {seed}")
    print("Model:", model_path)

    # Build the algorithm exactly like the working evaluator,
    # then load the saved .d3 model weights.
    algo = build_and_load_algo(
        algo_name=algo_name,
        model_path=str(model_path),
        dataset=dataset,
        observations=observations,
        actions=actions,
    )

    env = CityLearnEnv(
        str(SCHEMA_PATH),
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

    rows = []

    for step in range(MAX_STEPS):
        prediction_input = np.asarray(
            obs,
            dtype=np.float32,
        )

        if prediction_input.ndim == 1:
            prediction_input = prediction_input.reshape(
                1,
                -1,
            )

        action = algo.predict(
            prediction_input
        )[0]

        action = np.clip(
            action,
            action_low,
            action_high,
        )

        step_output = env.step([action])

        # Handle both 4-value and 5-value step APIs.
        if len(step_output) == 5:
            (
                next_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = step_output

            terminated_flag = bool(
                np.any(terminated)
            )

            truncated_flag = bool(
                np.any(truncated)
            )

            done = (
                terminated_flag
                or truncated_flag
            )

        elif len(step_output) == 4:
            (
                next_obs,
                reward,
                done,
                info,
            ) = step_output

            done = bool(
                np.any(done)
            )

        else:
            raise RuntimeError(
                "Unexpected CityLearn env.step() output."
            )

        # Record current net electricity from each building.
        building_values = []

        for building_id, building in enumerate(
            env.buildings
        ):
            values = np.asarray(
                building.net_electricity_consumption,
                dtype=np.float64,
            )

            current_index = max(0, env.time_step - 1)

            current_value = float(
              values[current_index]
            )

            building_values.append(
                current_value
            )

            rows.append(
                {
                    "algorithm": algo_name.upper(),
                    "seed": seed,
                    "step": step + 1,
                    "level": "building",
                    "building": building_id + 1,
                    "net_electricity": current_value,
                }
            )

        # Sum all three buildings for district-level value.
        district_value = float(
            np.sum(building_values)
        )

        rows.append(
            {
                "algorithm": algo_name.upper(),
                "seed": seed,
                "step": step + 1,
                "level": "district",
                "building": 0,
                "net_electricity": district_value,
            }
        )

        obs = obs_to_array(
            next_obs
        )

        if done:
            break

    env.close()

    print(
        "Completed steps:",
        step + 1,
    )

    return rows


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset, observations, actions = load_dataset(
        DATA_PATH
    )

    all_rows = []

    for algo_name in ALGORITHMS:
        for seed in SEEDS:
            rows = evaluate_timeseries(
                algo_name=algo_name,
                seed=seed,
                dataset=dataset,
                observations=observations,
                actions=actions,
            )

            all_rows.extend(
                rows
            )

    df = pd.DataFrame(
        all_rows
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("Saved:")
    print(OUTPUT_PATH)

    district_df = df[
        df["level"] == "district"
    ]

    print()
    print(
        "District rows:",
        len(district_df),
    )

    print()
    print(
        district_df.groupby(
            ["algorithm", "seed"]
        ).size()
    )

    print()
    print(
        "Mean district net electricity per timestep:"
    )

    print(
        district_df.groupby(
            "algorithm"
        )["net_electricity"].mean()
    )


if __name__ == "__main__":
    main()