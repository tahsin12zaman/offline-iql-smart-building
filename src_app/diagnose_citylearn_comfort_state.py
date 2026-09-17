from pathlib import Path
import csv
import sys

import numpy as np


# ---------------------------------------------------------------------
# Project / vendored CityLearn import path
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CITYLEARN_ROOT = PROJECT_ROOT / "external" / "CityLearn"

if str(CITYLEARN_ROOT) not in sys.path:
    sys.path.insert(0, str(CITYLEARN_ROOT))


from citylearn.citylearn import CityLearnEnv
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import IQLConfig


# ---------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------



SCHEMA = (
    PROJECT_ROOT
    / "external"
    / "CityLearn"
    / "data"
    / "datasets"
    / "citylearn_challenge_2023_phase_2_local_evaluation"
    / "schema.json"
)

DATA = PROJECT_ROOT / "data" / "citylearn_logged_multi.npz"

MODEL = PROJECT_ROOT / "results" / "citylearn_iql_seed_1.d3"

OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "citylearn_comfort_state_diagnostic.csv"
)

MAX_STEPS = 719


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def obs_to_array(obs):
    """Convert CityLearn observation to a flat float32 array."""

    if isinstance(obs, tuple):
        obs = obs[0]

    if isinstance(obs, list):
        obs = obs[0]

    return np.asarray(obs, dtype=np.float32).reshape(-1)


def reward_to_float(reward):
    """Convert CityLearn reward to one scalar."""

    if isinstance(reward, list):
        return float(np.sum(reward))

    return float(np.asarray(reward).sum())


def build_iql():
    """Build IQL using the same configuration as the existing evaluator."""

    return IQLConfig(
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        expectile=0.7,
        weight_temp=3.0,
        max_weight=100.0,
    ).create(device=False)


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_dataset():
    """Load the existing CityLearn offline dataset."""

    arr = np.load(DATA, allow_pickle=True)

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


def load_model():
    """Load the already-trained IQL Seed 1 controller."""

    dataset, observations, actions = load_dataset()

    algo = build_iql()

    try:
        algo.build_with_dataset(dataset)

    except AttributeError:
        algo.create_impl(
            observation_shape=observations.shape[1:],
            action_size=actions.shape[1],
        )

    if not MODEL.exists():
        raise FileNotFoundError(
            f"Saved IQL model not found: {MODEL}"
        )

    algo.load_model(str(MODEL))

    return algo


# ---------------------------------------------------------------------
# Runtime-state diagnostic
# ---------------------------------------------------------------------

def get_series(building, attribute_name):
    """Return a building attribute as a flat float64 array."""

    values = getattr(building, attribute_name)

    return np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)


def inspect_runtime_value(
    building,
    attribute_name,
):
    """
    Inspect several plausible indices for a CityLearn time series.

    We deliberately record:
        - building.time_step
        - value at building.time_step
        - value at building.time_step - 1
        - final element of the array

    This lets us diagnose CityLearn's runtime/history alignment instead
    of assuming which index represents the currently observable state.
    """

    arr = get_series(
        building,
        attribute_name,
    )

    time_step = int(building.time_step)

    value_at_time_step = np.nan
    value_at_previous_step = np.nan
    value_at_last_index = np.nan

    if 0 <= time_step < arr.size:
        value_at_time_step = float(
            arr[time_step]
        )

    previous_index = time_step - 1

    if 0 <= previous_index < arr.size:
        value_at_previous_step = float(
            arr[previous_index]
        )

    if arr.size > 0:
        value_at_last_index = float(
            arr[-1]
        )

    return {
        "array_length": int(arr.size),
        "time_step": time_step,
        "at_time_step": value_at_time_step,
        "at_previous_step": value_at_previous_step,
        "at_last_index": value_at_last_index,
    }


def violation(
    indoor_temperature,
    cooling_setpoint,
    comfort_band,
    occupant_count,
):
    """
    Apply the exact overheating definition used by the project:

        occupied
        AND
        indoor_temperature >
            cooling_setpoint + comfort_band
    """

    values = [
        indoor_temperature,
        cooling_setpoint,
        comfort_band,
        occupant_count,
    ]

    if not all(np.isfinite(v) for v in values):
        return False

    occupied = occupant_count > 0.0

    upper_boundary = (
        cooling_setpoint
        + comfort_band
    )

    return bool(
        occupied
        and indoor_temperature > upper_boundary
    )


# ---------------------------------------------------------------------
# Completed-history calculation
# ---------------------------------------------------------------------

def calculate_history_violations(building):
    """
    Reproduce the existing final-history overheating calculation.

    Returns:
        violation_count
        occupied_count
        discomfort_fraction
    """

    indoor = get_series(
        building,
        "indoor_dry_bulb_temperature",
    )

    cooling_setpoint = get_series(
        building,
        "indoor_dry_bulb_temperature_cooling_set_point",
    )

    comfort_band = get_series(
        building,
        "comfort_band",
    )

    occupant_count = get_series(
        building,
        "occupant_count",
    )

    n = min(
        len(indoor),
        len(cooling_setpoint),
        len(comfort_band),
        len(occupant_count),
    )

    indoor = indoor[:n]
    cooling_setpoint = cooling_setpoint[:n]
    comfort_band = comfort_band[:n]
    occupant_count = occupant_count[:n]

    occupied = occupant_count > 0.0

    upper_boundary = (
        cooling_setpoint
        + comfort_band
    )

    overheated = (
        occupied
        & (indoor > upper_boundary)
    )

    occupied_count = int(
        np.sum(occupied)
    )

    violation_count = int(
        np.sum(overheated)
    )

    if occupied_count == 0:
        discomfort = float("nan")
    else:
        discomfort = (
            violation_count
            / occupied_count
        )

    return {
        "samples": n,
        "occupied_count": occupied_count,
        "violation_count": violation_count,
        "discomfort": discomfort,
    }


# ---------------------------------------------------------------------
# Main diagnostic rollout
# ---------------------------------------------------------------------

def main():

    print("=" * 78)
    print("CityLearn Runtime Comfort-State Diagnostic")
    print("=" * 78)

    print(f"Schema: {SCHEMA}")
    print(f"Dataset: {DATA}")
    print(f"Model:   {MODEL}")
    print()

    print("Loading IQL Seed 1 controller...")

    algo = load_model()

    print("Controller loaded.")
    print()

    env = CityLearnEnv(
        str(SCHEMA),
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

    obs = obs_to_array(
        env.reset()
    )

    rows = []

    runtime_counts_time_step = {
        i: 0
        for i in range(1, len(env.buildings) + 1)
    }

    runtime_counts_previous = {
        i: 0
        for i in range(1, len(env.buildings) + 1)
    }

    runtime_counts_last = {
        i: 0
        for i in range(1, len(env.buildings) + 1)
    }

    total_reward = 0.0
    steps = 0

    print("Running ordinary IQL rollout...")
    print()

    for loop_step in range(MAX_STEPS):

        # -------------------------------------------------------------
        # Diagnose CityLearn state BEFORE taking this control action.
        # -------------------------------------------------------------

        for building_index, building in enumerate(
            env.buildings,
            start=1,
        ):

            indoor = inspect_runtime_value(
                building,
                "indoor_dry_bulb_temperature",
            )

            setpoint = inspect_runtime_value(
                building,
                "indoor_dry_bulb_temperature_cooling_set_point",
            )

            band = inspect_runtime_value(
                building,
                "comfort_band",
            )

            occupants = inspect_runtime_value(
                building,
                "occupant_count",
            )

            violation_at_time_step = violation(
                indoor["at_time_step"],
                setpoint["at_time_step"],
                band["at_time_step"],
                occupants["at_time_step"],
            )

            violation_at_previous = violation(
                indoor["at_previous_step"],
                setpoint["at_previous_step"],
                band["at_previous_step"],
                occupants["at_previous_step"],
            )

            violation_at_last = violation(
                indoor["at_last_index"],
                setpoint["at_last_index"],
                band["at_last_index"],
                occupants["at_last_index"],
            )

            if violation_at_time_step:
                runtime_counts_time_step[
                    building_index
                ] += 1

            if violation_at_previous:
                runtime_counts_previous[
                    building_index
                ] += 1

            if violation_at_last:
                runtime_counts_last[
                    building_index
                ] += 1

            upper_at_time_step = (
                setpoint["at_time_step"]
                + band["at_time_step"]
                if (
                    np.isfinite(
                        setpoint["at_time_step"]
                    )
                    and np.isfinite(
                        band["at_time_step"]
                    )
                )
                else np.nan
            )

            upper_at_previous = (
                setpoint["at_previous_step"]
                + band["at_previous_step"]
                if (
                    np.isfinite(
                        setpoint["at_previous_step"]
                    )
                    and np.isfinite(
                        band["at_previous_step"]
                    )
                )
                else np.nan
            )

            upper_at_last = (
                setpoint["at_last_index"]
                + band["at_last_index"]
                if (
                    np.isfinite(
                        setpoint["at_last_index"]
                    )
                    and np.isfinite(
                        band["at_last_index"]
                    )
                )
                else np.nan
            )

            rows.append(
                {
                    "loop_step": loop_step,
                    "building": building_index,
                    "building_time_step":
                        indoor["time_step"],
                    "series_length":
                        indoor["array_length"],

                    "indoor_at_time_step":
                        indoor["at_time_step"],
                    "setpoint_at_time_step":
                        setpoint["at_time_step"],
                    "comfort_band_at_time_step":
                        band["at_time_step"],
                    "occupants_at_time_step":
                        occupants["at_time_step"],
                    "upper_boundary_at_time_step":
                        upper_at_time_step,
                    "violation_at_time_step":
                        int(violation_at_time_step),

                    "indoor_at_previous":
                        indoor["at_previous_step"],
                    "setpoint_at_previous":
                        setpoint["at_previous_step"],
                    "comfort_band_at_previous":
                        band["at_previous_step"],
                    "occupants_at_previous":
                        occupants["at_previous_step"],
                    "upper_boundary_at_previous":
                        upper_at_previous,
                    "violation_at_previous":
                        int(violation_at_previous),

                    "indoor_at_last":
                        indoor["at_last_index"],
                    "setpoint_at_last":
                        setpoint["at_last_index"],
                    "comfort_band_at_last":
                        band["at_last_index"],
                    "occupants_at_last":
                        occupants["at_last_index"],
                    "upper_boundary_at_last":
                        upper_at_last,
                    "violation_at_last":
                        int(violation_at_last),
                }
            )

        # -------------------------------------------------------------
        # Ordinary saved-policy action.
        # NO GUARDRAIL.
        # NO RBC.
        # -------------------------------------------------------------

        action = algo.predict(
            obs.reshape(1, -1)
        )[0].astype(np.float32)

        action = np.clip(
            action,
            action_low,
            action_high,
        )

        step_out = env.step(
            [action]
        )

        if len(step_out) == 5:
            (
                next_obs,
                reward,
                terminated,
                truncated,
                _,
            ) = step_out

            done = bool(
                terminated or truncated
            )

        else:
            (
                next_obs,
                reward,
                done,
                _,
            ) = step_out

            done = bool(done)

        total_reward += reward_to_float(
            reward
        )

        steps += 1

        obs = obs_to_array(
            next_obs
        )

        if done:
            break

    # -----------------------------------------------------------------
    # Save raw runtime diagnostic
    # -----------------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if rows:
        with OUTPUT.open(
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(rows)

    # -----------------------------------------------------------------
    # Compare runtime detection with completed histories
    # -----------------------------------------------------------------

    print("=" * 78)
    print("ROLLOUT COMPLETE")
    print("=" * 78)

    print(
        f"Control steps:      {steps}"
    )

    print(
        f"Cumulative reward:  {total_reward:.6f}"
    )

    print()

    print("=" * 78)
    print("RUNTIME vs COMPLETED-HISTORY COMPARISON")
    print("=" * 78)

    for building_index, building in enumerate(
        env.buildings,
        start=1,
    ):

        history = calculate_history_violations(
            building
        )

        print()
        print(
            f"BUILDING {building_index}"
        )

        print("-" * 78)

        print(
            "Runtime violations using "
            "building.time_step:      "
            f"{runtime_counts_time_step[building_index]}"
        )

        print(
            "Runtime violations using "
            "time_step - 1:           "
            f"{runtime_counts_previous[building_index]}"
        )

        print(
            "Runtime violations using "
            "array[-1]:               "
            f"{runtime_counts_last[building_index]}"
        )

        print(
            "Post-rollout violations: "
            f"{history['violation_count']}"
        )

        print(
            "Post-rollout occupied:   "
            f"{history['occupied_count']}"
        )

        if np.isfinite(
            history["discomfort"]
        ):
            print(
                "Post-rollout discomfort: "
                f"{100.0 * history['discomfort']:.2f}%"
            )

        else:
            print(
                "Post-rollout discomfort: NaN"
            )

    # -----------------------------------------------------------------
    # Show first few rows where any indexing interpretation disagrees
    # -----------------------------------------------------------------

    disagreements = [
        row
        for row in rows
        if not (
            row["violation_at_time_step"]
            == row["violation_at_previous"]
            == row["violation_at_last"]
        )
    ]

    print()
    print("=" * 78)
    print("INDEXING DISAGREEMENTS")
    print("=" * 78)

    print(
        f"Rows with disagreement: {len(disagreements)}"
    )

    if disagreements:

        print()
        print(
            "First 15 disagreement rows:"
        )

        for row in disagreements[:15]:

            print(
                f"loop={row['loop_step']:3d} "
                f"B={row['building']} "
                f"citylearn_t={row['building_time_step']:3d} "
                f"len={row['series_length']:3d} | "
                f"t={row['violation_at_time_step']} "
                f"t-1={row['violation_at_previous']} "
                f"last={row['violation_at_last']} | "
                f"T(t)={row['indoor_at_time_step']:.3f} "
                f"upper(t)={row['upper_boundary_at_time_step']:.3f} "
                f"occ(t)={row['occupants_at_time_step']:.3f}"
            )

    else:

        print(
            "No disagreement between the three "
            "runtime indexing interpretations."
        )

    print()
    print("=" * 78)
    print("OUTPUT")
    print("=" * 78)

    print(
        f"Diagnostic CSV: {OUTPUT}"
    )

    print()

    env.close()


if __name__ == "__main__":
    main()