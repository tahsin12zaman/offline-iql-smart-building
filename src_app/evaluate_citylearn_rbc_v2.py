from pathlib import Path
import argparse
import csv
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CITYLEARN_ROOT = PROJECT_ROOT / "external" / "CityLearn"

if str(CITYLEARN_ROOT) not in sys.path:
    sys.path.insert(0, str(CITYLEARN_ROOT))

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.rbc import BasicRBC


DEFAULT_SCHEMA = (
    PROJECT_ROOT
    / "external"
    / "CityLearn"
    / "data"
    / "datasets"
    / "citylearn_challenge_2023_phase_2_local_evaluation"
    / "schema.json"
)


def series(building, name):
    return np.asarray(
        getattr(building, name),
        dtype=np.float64,
    ).reshape(-1)


def reward_to_float(reward):
    return float(np.asarray(reward).sum())


def sum_metric(env, name):
    return sum(
        float(np.nansum(series(building, name)))
        for building in env.buildings
    )


def peak_metric(env, name):
    arrays = [
        series(building, name)
        for building in env.buildings
    ]
    arrays = [array for array in arrays if array.size]

    if not arrays:
        return float("nan")

    return float(np.nanmax(np.concatenate(arrays)))


def calculate_building_comfort(building):
    """
    Calculate comfort using the exact same definition used by
    evaluate_citylearn_guarded_v2.py.

    Overheating:
        occupied AND indoor > setpoint + comfort_band

    Overcooling:
        occupied AND indoor < setpoint - comfort_band

    Combined discomfort:
        overheating OR overcooling
    """

    indoor = series(
        building,
        "indoor_dry_bulb_temperature",
    )

    setpoint = series(
        building,
        "indoor_dry_bulb_temperature_cooling_set_point",
    )

    comfort_band = series(
        building,
        "comfort_band",
    )

    occupants = series(
        building,
        "occupant_count",
    )

    n = min(
        len(indoor),
        len(setpoint),
        len(comfort_band),
        len(occupants),
    )

    if n == 0:
        return {
            "occupied_count": 0,
            "violation_count": 0,
            "discomfort": float("nan"),
            "overcooling_violation_count": 0,
            "overcooling_discomfort": float("nan"),
            "within_band_cool_count": 0,
            "within_band_cool_rate": float("nan"),
            "combined_violation_count": 0,
            "combined_discomfort": float("nan"),
        }

    indoor = indoor[:n]
    setpoint = setpoint[:n]
    comfort_band = comfort_band[:n]
    occupants = occupants[:n]

    occupied = occupants > 0.0

    lower = setpoint - comfort_band
    upper = setpoint + comfort_band

    # Occupied and above upper comfort boundary.
    overheating = occupied & (indoor > upper)

    # Occupied and below lower comfort boundary.
    overcooling = occupied & (indoor < lower)

    # Occupied and below setpoint, but still inside comfort band.
    within_band_cool = (
        occupied
        & (indoor < setpoint)
        & (indoor >= lower)
    )

    # Outside the comfort band in either direction.
    combined = overheating | overcooling

    occupied_count = int(np.sum(occupied))

    def rate(mask):
        return (
            float(np.sum(mask) / occupied_count)
            if occupied_count > 0
            else float("nan")
        )

    return {
        # Same compatibility fields as guarded-IQL evaluator.
        "occupied_count": occupied_count,
        "violation_count": int(np.sum(overheating)),
        "discomfort": rate(overheating),

        "overcooling_violation_count": int(
            np.sum(overcooling)
        ),
        "overcooling_discomfort": rate(overcooling),

        "within_band_cool_count": int(
            np.sum(within_band_cool)
        ),
        "within_band_cool_rate": rate(
            within_band_cool
        ),

        "combined_violation_count": int(
            np.sum(combined)
        ),
        "combined_discomfort": rate(combined),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--schema",
        default=str(DEFAULT_SCHEMA),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=719,
    )

    parser.add_argument(
        "--out",
        default="results/original_weather_rbc_summary.csv",
    )

    args = parser.parse_args()

    schema = Path(args.schema).resolve()

    env = CityLearnEnv(
        str(schema),
        central_agent=True,
    )

    try:
        controller = BasicRBC(env)

        action_space = env.action_space[0]

        low = np.asarray(
            action_space.low,
            dtype=np.float32,
        )

        high = np.asarray(
            action_space.high,
            dtype=np.float32,
        )

        reset_out = env.reset()

        obs = (
            reset_out[0]
            if isinstance(reset_out, tuple)
            else reset_out
        )

        total_reward = 0.0
        steps = 0

        for _ in range(args.max_steps):

            action = np.asarray(
                controller.predict(
                    obs,
                    deterministic=True,
                ),
                dtype=np.float32,
            ).reshape(-1)

            action = np.clip(
                action,
                low,
                high,
            )

            result = env.step([action])

            if len(result) == 5:
                (
                    next_obs,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = result

                done = bool(
                    np.any(terminated)
                    or np.any(truncated)
                )

            else:
                (
                    next_obs,
                    reward,
                    done,
                    _,
                ) = result

                done = bool(np.any(done))

            total_reward += reward_to_float(reward)
            steps += 1

            obs = (
                next_obs[0]
                if isinstance(next_obs, tuple)
                else next_obs
            )

            if done:
                break

        metrics = {
            "controller": "BasicRBC",
            "schema": str(schema),
            "steps": steps,
            "reward": total_reward,

            "net_electricity_consumption": sum_metric(
                env,
                "net_electricity_consumption",
            ),

            "electricity_cost": sum_metric(
                env,
                "net_electricity_consumption_cost",
            ),

            "carbon_emission": sum_metric(
                env,
                "net_electricity_consumption_emission",
            ),

            "peak_net_electricity": peak_metric(
                env,
                "net_electricity_consumption",
            ),
        }

        overheating_rates = []
        overcooling_rates = []
        combined_rates = []

        for i, building in enumerate(
            env.buildings,
            start=1,
        ):
            c = calculate_building_comfort(building)

            metrics[
                f"building_{i}_occupied_count"
            ] = c["occupied_count"]

            # Overheating
            metrics[
                f"building_{i}_violation_count"
            ] = c["violation_count"]

            metrics[
                f"building_{i}_overheating_discomfort"
            ] = c["discomfort"]

            # Overcooling
            metrics[
                f"building_{i}_overcooling_violation_count"
            ] = c["overcooling_violation_count"]

            metrics[
                f"building_{i}_overcooling_discomfort"
            ] = c["overcooling_discomfort"]

            # Within comfort band on cool side
            metrics[
                f"building_{i}_within_band_cool_count"
            ] = c["within_band_cool_count"]

            metrics[
                f"building_{i}_within_band_cool_rate"
            ] = c["within_band_cool_rate"]

            # Combined discomfort
            metrics[
                f"building_{i}_combined_violation_count"
            ] = c["combined_violation_count"]

            metrics[
                f"building_{i}_combined_discomfort"
            ] = c["combined_discomfort"]

            if np.isfinite(c["discomfort"]):
                overheating_rates.append(
                    c["discomfort"]
                )

            if np.isfinite(
                c["overcooling_discomfort"]
            ):
                overcooling_rates.append(
                    c["overcooling_discomfort"]
                )

            if np.isfinite(
                c["combined_discomfort"]
            ):
                combined_rates.append(
                    c["combined_discomfort"]
                )

        metrics[
            "worst_building_overheating_discomfort"
        ] = (
            float(max(overheating_rates))
            if overheating_rates
            else float("nan")
        )

        metrics[
            "worst_building_overcooling_discomfort"
        ] = (
            float(max(overcooling_rates))
            if overcooling_rates
            else float("nan")
        )

        metrics[
            "worst_building_combined_discomfort"
        ] = (
            float(max(combined_rates))
            if combined_rates
            else float("nan")
        )

        # ---------------------------------------------------------
        # Console output
        # ---------------------------------------------------------

        print("=" * 78)
        print("CityLearn BasicRBC Evaluation")
        print("=" * 78)

        print(f"Schema: {schema}")
        print(f"Control steps:             {steps}")
        print(
            f"Cumulative reward:         "
            f"{total_reward:.6f}"
        )
        print(
            "Net electricity:           "
            f"{metrics['net_electricity_consumption']:.6f}"
        )
        print(
            "Electricity cost:          "
            f"{metrics['electricity_cost']:.6f}"
        )
        print(
            "Carbon emissions:          "
            f"{metrics['carbon_emission']:.6f}"
        )
        print(
            "Peak building electricity: "
            f"{metrics['peak_net_electricity']:.6f}"
        )

        print("\nComfort metrics")
        print("-" * 78)

        for i in range(
            1,
            len(env.buildings) + 1,
        ):
            occupied = metrics[
                f"building_{i}_occupied_count"
            ]

            overheating_count = metrics[
                f"building_{i}_violation_count"
            ]

            overcooling_count = metrics[
                f"building_{i}_overcooling_violation_count"
            ]

            combined_count = metrics[
                f"building_{i}_combined_violation_count"
            ]

            overheating = metrics[
                f"building_{i}_overheating_discomfort"
            ]

            overcooling = metrics[
                f"building_{i}_overcooling_discomfort"
            ]

            combined = metrics[
                f"building_{i}_combined_discomfort"
            ]

            overheating_text = (
                f"{100 * overheating:.2f}%"
                if np.isfinite(overheating)
                else "NaN"
            )

            overcooling_text = (
                f"{100 * overcooling:.2f}%"
                if np.isfinite(overcooling)
                else "NaN"
            )

            combined_text = (
                f"{100 * combined:.2f}%"
                if np.isfinite(combined)
                else "NaN"
            )

            print(
                f"Building {i}:"
            )
            print(
                f"  overheating: "
                f"{overheating_text:>8s} "
                f"({overheating_count}/{occupied})"
            )
            print(
                f"  overcooling: "
                f"{overcooling_text:>8s} "
                f"({overcooling_count}/{occupied})"
            )
            print(
                f"  combined:    "
                f"{combined_text:>8s} "
                f"({combined_count}/{occupied})"
            )

        worst_overheating = metrics[
            "worst_building_overheating_discomfort"
        ]

        worst_overcooling = metrics[
            "worst_building_overcooling_discomfort"
        ]

        worst_combined = metrics[
            "worst_building_combined_discomfort"
        ]

        print("\nWorst-building comfort")
        print("-" * 78)

        if np.isfinite(worst_overheating):
            print(
                "Worst overheating: "
                f"{100 * worst_overheating:.2f}%"
            )

        if np.isfinite(worst_overcooling):
            print(
                "Worst overcooling:  "
                f"{100 * worst_overcooling:.2f}%"
            )

        if np.isfinite(worst_combined):
            print(
                "Worst combined:     "
                f"{100 * worst_combined:.2f}%"
            )

        # ---------------------------------------------------------
        # Save CSV
        # ---------------------------------------------------------

        out = Path(args.out)

        out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with out.open(
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(metrics.keys()),
            )

            writer.writeheader()
            writer.writerow(metrics)

        print(f"\nSummary saved to: {out}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
