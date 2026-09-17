
from pathlib import Path
import sys
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CITYLEARN_ROOT = PROJECT_ROOT / "external" / "CityLearn"

if str(CITYLEARN_ROOT) not in sys.path:
    sys.path.insert(0, str(CITYLEARN_ROOT))

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.rbc import BasicRBC


SCHEMA = (
    PROJECT_ROOT
    / "external"
    / "CityLearn"
    / "data"
    / "datasets"
    / "citylearn_challenge_2023_phase_2_local_evaluation"
    / "schema.json"
)


def reward_to_float(reward):
    if isinstance(reward, list):
        return float(np.sum(reward))

    return float(np.asarray(reward).sum())


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


def main():
    print("=" * 70)
    print("CityLearn Conventional Baseline Evaluation: BasicRBC")
    print("=" * 70)
    print(f"Schema: {SCHEMA}")

    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=True,
    )

    controller = BasicRBC(env)

    observations = env.reset()

    if isinstance(observations, tuple):
        observations = observations[0]

    cumulative_reward = 0.0
    steps = 0

    while not env.terminated:
        actions = controller.predict(
            observations,
            deterministic=True,
        )

        step_result = env.step(actions)

        if len(step_result) == 5:
            observations, reward, terminated, truncated, _ = step_result
            done = bool(terminated or truncated)
        else:
            observations, reward, done, _ = step_result
            done = bool(done)

        cumulative_reward += reward_to_float(reward)
        steps += 1

        if done:
            break

    metrics = {
        "reward": cumulative_reward,
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

    print()
    print("Evaluation complete.")
    print("-" * 70)
    print("Controller:               BasicRBC")
    print(f"Control steps:            {metrics['steps']}")
    print(f"Cumulative reward:        {metrics['reward']:.6f}")
    print(
        "Net electricity:          "
        f"{metrics['net_electricity_consumption']:.6f}"
    )
    print(
        "Electricity cost:         "
        f"{metrics['electricity_cost']:.6f}"
    )
    print(
        "Carbon emissions:         "
        f"{metrics['carbon_emission']:.6f}"
    )
    print(
        "Peak building electricity:"
        f" {metrics['peak_net_electricity']:.6f}"
    )
    print("-" * 70)

    print()
    print("Per-building net electricity:")

    for i, building in enumerate(env.buildings, start=1):
        value = float(
            np.nansum(
                np.asarray(
                    building.net_electricity_consumption,
                    dtype=np.float64,
                )
            )
        )

        peak = float(
            np.nanmax(
                np.asarray(
                    building.net_electricity_consumption,
                    dtype=np.float64,
                )
            )
        )

        print(
            f"  Building {i}: "
            f"total={value:.6f}, peak={peak:.6f}"
        )

    env.close()


if __name__ == "__main__":
    main()
