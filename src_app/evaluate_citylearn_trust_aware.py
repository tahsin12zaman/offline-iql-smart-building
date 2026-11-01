from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CITYLEARN_ROOT = PROJECT_ROOT / "external" / "CityLearn"

if str(CITYLEARN_ROOT) not in sys.path:
    sys.path.insert(0, str(CITYLEARN_ROOT))

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.rbc import BasicRBC

from evaluate_citylearn_guarded_v2 import (
    load_iql_model,
    obs_to_array,
    reward_to_float,
    collect_metrics,
    get_completed_comfort_state,
    take_step,
)


DEFAULT_DATA = (
    PROJECT_ROOT
    / "data"
    / "citylearn_logged_multi.npz"
)

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "results"
    / "citylearn_iql_seed_1.d3"
)

DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "results"
    / "citylearn_time_conditioned_reference.npz"
)

ORIGINAL_SCHEMA = (
    PROJECT_ROOT
    / "external"
    / "CityLearn"
    / "data"
    / "datasets"
    / "citylearn_challenge_2023_phase_2_local_evaluation"
    / "schema.json"
)

SCENARIOS = [
    (
        "Original",
        0.0,
        ORIGINAL_SCHEMA,
    ),
    (
        "+1C",
        1.0,
        PROJECT_ROOT
        / "data"
        / "scenarios"
        / "hot_weather_plus1C"
        / "schema.json",
    ),
    (
        "+2C",
        2.0,
        PROJECT_ROOT
        / "data"
        / "scenarios"
        / "hot_weather_plus2C"
        / "schema.json",
    ),
    (
        "+3C",
        3.0,
        PROJECT_ROOT
        / "data"
        / "scenarios"
        / "hot_weather_plus3C"
        / "schema.json",
    ),
]


CONTROLLERS = [
    "iql",
    "guarded",
    "ood_only",
    "trust_aware",
    "trust_aware_guarded",
]


DISPLAY_NAMES = {
    "iql": "IQL",
    "guarded": "Comfort-Guarded IQL",
    "ood_only": "OOD-Gated IQL",
    "trust_aware": "Trust-Aware IQL",
    "trust_aware_guarded":
        "Trust-Aware + Guardrail",
}


def load_trust_reference(path, threshold_name):
    ref = np.load(
        path,
        allow_pickle=True,
    )

    time_mean = ref[
        "time_mean"
    ].astype(np.float32)

    safe_time_std = ref[
        "safe_time_std"
    ].astype(np.float32)

    active_mask = ref[
        "active_feature_mask"
    ].astype(bool)

    if threshold_name == "p95":
        threshold = float(
            ref["threshold_95"]
        )

    elif threshold_name == "p99":
        threshold = float(
            ref["threshold_99"]
        )

    else:
        raise ValueError(
            f"Unknown threshold: {threshold_name}"
        )

    return (
        time_mean,
        safe_time_std,
        active_mask,
        threshold,
    )


def calculate_ood_score(
    obs,
    timestep,
    time_mean,
    safe_time_std,
    active_mask,
):
    if timestep >= len(time_mean):
        timestep = len(time_mean) - 1

    z = (
        obs - time_mean[timestep]
    ) / safe_time_std[timestep]

    z = z[
        active_mask
    ]

    return float(
        np.sqrt(
            np.mean(
                np.square(z)
            )
        )
    )


def get_rbc_observations(value):
    if isinstance(value, tuple):
        return value[0]

    return value


def determine_trigger(
    controller,
    low_trust,
    state,
    risk_margin,
):
    if state is None:
        return False, None

    occupied = (
        state["occupant_count"] > 0.0
    )

    violation = bool(
        state["violation"]
    )

    risk_boundary = (
        state[
            "upper_comfort_boundary"
        ]
        - risk_margin
    )

    thermal_risk = bool(
        occupied
        and
        state[
            "indoor_temperature"
        ] > risk_boundary
    )

    if controller == "iql":
        return False, None

    if controller == "guarded":
        if violation:
            return (
                True,
                "comfort_violation",
            )

        return False, None

    if controller == "ood_only":
        if low_trust:
            return (
                True,
                "low_trust",
            )

        return False, None

    if controller == "trust_aware":
        if (
            low_trust
            and thermal_risk
        ):
            if violation:
                reason = (
                    "low_trust_and_violation"
                )
            else:
                reason = (
                    "low_trust_and_thermal_risk"
                )

            return True, reason

        return False, None

    if controller == "trust_aware_guarded":

        if violation:
            if low_trust:
                reason = (
                    "low_trust_and_violation"
                )
            else:
                reason = (
                    "hard_guardrail_violation"
                )

            return True, reason

        if (
            low_trust
            and thermal_risk
        ):
            return (
                True,
                "low_trust_and_thermal_risk",
            )

        return False, None

    raise ValueError(
        f"Unknown controller: {controller}"
    )


def evaluate_controller(
    algo,
    schema,
    scenario,
    shift_c,
    controller,
    max_steps,
    time_mean,
    safe_time_std,
    active_mask,
    ood_threshold,
    threshold_name,
    risk_margin,
):
    env = CityLearnEnv(
        str(schema),
        central_agent=True,
    )

    rbc = BasicRBC(
        env
    )

    try:
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

        obs = obs_to_array(
            reset_out
        )

        observations_for_rbc = (
            get_rbc_observations(
                reset_out
            )
        )

        cooling_indices = [
            i
            for i, name
            in enumerate(
                env.action_names[0]
            )
            if name == "cooling_device"
        ]

        if (
            len(cooling_indices)
            != len(env.buildings)
        ):
            raise RuntimeError(
                "Expected one cooling_device "
                "action per building. "
                f"Found {cooling_indices}"
            )

        total_reward = 0.0
        steps = 0

        intervention_count = 0
        low_trust_steps = 0

        intervention_by_building = {
            i: 0
            for i in range(
                1,
                len(env.buildings) + 1,
            )
        }

        events = []
        trace = []

        for step in range(max_steps):

            ood_score = calculate_ood_score(
                obs,
                step,
                time_mean,
                safe_time_std,
                active_mask,
            )

            low_trust = bool(
                ood_score
                > ood_threshold
            )

            if low_trust:
                low_trust_steps += 1

            learned_action = (
                algo.predict(
                    obs.reshape(1, -1)
                )[0]
                .astype(np.float32)
            )

            learned_action = np.clip(
                learned_action,
                low,
                high,
            )

            action = (
                learned_action.copy()
            )

            rbc_raw = rbc.predict(
                observations_for_rbc,
                deterministic=True,
            )

            rbc_action = np.asarray(
                rbc_raw,
                dtype=np.float32,
            ).reshape(-1)

            if (
                rbc_action.shape
                != action.shape
            ):
                raise RuntimeError(
                    "RBC/IQL action shape "
                    "mismatch: "
                    f"RBC={rbc_action.shape}, "
                    f"IQL={action.shape}"
                )

            rbc_action = np.clip(
                rbc_action,
                low,
                high,
            )

            step_interventions = 0

            for (
                building_index,
                building,
            ) in enumerate(
                env.buildings,
                start=1,
            ):

                state = (
                    get_completed_comfort_state(
                        building
                    )
                )

                trigger, reason = (
                    determine_trigger(
                        controller,
                        low_trust,
                        state,
                        risk_margin,
                    )
                )

                if not trigger:
                    continue

                cooling_index = (
                    cooling_indices[
                        building_index - 1
                    ]
                )

                learned_cooling = float(
                    action[
                        cooling_index
                    ]
                )

                fallback_cooling = float(
                    rbc_action[
                        cooling_index
                    ]
                )

                action[
                    cooling_index
                ] = fallback_cooling

                intervention_count += 1
                step_interventions += 1

                intervention_by_building[
                    building_index
                ] += 1

                event = {
                    "scenario":
                        scenario,
                    "shift_c":
                        shift_c,
                    "controller":
                        DISPLAY_NAMES[
                            controller
                        ],
                    "controller_key":
                        controller,
                    "step":
                        step,
                    "building":
                        building_index,
                    "ood_score":
                        ood_score,
                    "ood_threshold":
                        ood_threshold,
                    "low_trust":
                        low_trust,
                    "reason":
                        reason,
                    "risk_margin":
                        risk_margin,
                    "learned_cooling_action":
                        learned_cooling,
                    "fallback_cooling_action":
                        fallback_cooling,
                    "action_difference":
                        (
                            fallback_cooling
                            - learned_cooling
                        ),
                }

                if state is not None:
                    event.update({
                        "completed_index":
                            state[
                                "completed_index"
                            ],
                        "indoor_temperature":
                            state[
                                "indoor_temperature"
                            ],
                        "cooling_setpoint":
                            state[
                                "cooling_setpoint"
                            ],
                        "comfort_band":
                            state[
                                "comfort_band"
                            ],
                        "upper_comfort_boundary":
                            state[
                                "upper_comfort_boundary"
                            ],
                        "occupant_count":
                            state[
                                "occupant_count"
                            ],
                        "comfort_violation":
                            state[
                                "violation"
                            ],
                    })

                events.append(
                    event
                )

            action = np.clip(
                action,
                low,
                high,
            )

            trace.append({
                "scenario":
                    scenario,
                "shift_c":
                    shift_c,
                "controller":
                    DISPLAY_NAMES[
                        controller
                    ],
                "controller_key":
                    controller,
                "step":
                    step,
                "ood_score":
                    ood_score,
                "ood_threshold":
                    ood_threshold,
                "low_trust":
                    low_trust,
                "interventions":
                    step_interventions,
            })

            (
                next_obs,
                reward,
                done,
                _,
            ) = take_step(
                env,
                action,
            )

            total_reward += (
                reward_to_float(
                    reward
                )
            )

            steps += 1

            observations_for_rbc = (
                get_rbc_observations(
                    next_obs
                )
            )

            obs = obs_to_array(
                next_obs
            )

            if done:
                break

        metrics = collect_metrics(
            env,
            total_reward,
            steps,
        )

        metrics.update({
            "scenario":
                scenario,
            "shift_c":
                shift_c,
            "controller":
                DISPLAY_NAMES[
                    controller
                ],
            "controller_key":
                controller,
            "ood_threshold_type":
                threshold_name,
            "ood_threshold":
                ood_threshold,
            "risk_margin_c":
                risk_margin,
            "low_trust_steps":
                low_trust_steps,
            "low_trust_fraction":
                (
                    low_trust_steps
                    / steps
                    if steps
                    else float("nan")
                ),
            "intervention_count":
                intervention_count,
        })

        for building_index, count in (
            intervention_by_building.items()
        ):
            metrics[
                f"building_{building_index}"
                "_intervention_count"
            ] = count

        return (
            metrics,
            events,
            trace,
        )

    finally:
        env.close()


def print_run_result(metrics):
    print(
        f"Reward:        "
        f"{metrics['reward']:.6f}"
    )

    print(
        f"Energy:        "
        f"{metrics['net_electricity_consumption']:.6f}"
    )

    print(
        f"Cost:          "
        f"{metrics['electricity_cost']:.6f}"
    )

    print(
        f"Carbon:        "
        f"{metrics['carbon_emission']:.6f}"
    )

    print(
        f"Peak:          "
        f"{metrics['peak_net_electricity']:.6f}"
    )

    print(
        "Worst overheat:"
        f" {100 * metrics['worst_building_overheating_discomfort']:.2f}%"
    )

    print(
        "Worst overcool:"
        f" {100 * metrics['worst_building_overcooling_discomfort']:.2f}%"
    )

    print(
        "Worst combined:"
        f" {100 * metrics['worst_building_combined_discomfort']:.2f}%"
    )

    print(
        f"Low-trust:     "
        f"{100 * metrics['low_trust_fraction']:.2f}%"
    )

    print(
        f"Interventions: "
        f"{metrics['intervention_count']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trust-aware CityLearn "
            "IQL controllers under temperature "
            "distribution shift."
        )
    )

    parser.add_argument(
        "--data",
        default=str(
            DEFAULT_DATA
        ),
    )

    parser.add_argument(
        "--model",
        default=str(
            DEFAULT_MODEL
        ),
    )

    parser.add_argument(
        "--reference",
        default=str(
            DEFAULT_REFERENCE
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=719,
    )

    parser.add_argument(
        "--threshold",
        choices=[
            "p95",
            "p99",
        ],
        default="p95",
    )

    parser.add_argument(
        "--risk-margin",
        type=float,
        default=0.5,
        help=(
            "Degrees C below upper comfort "
            "boundary at which an occupied "
            "building is considered thermally "
            "at risk."
        ),
    )

    parser.add_argument(
        "--summary-out",
        default=(
            "results/"
            "citylearn_trust_aware_summary.csv"
        ),
    )

    parser.add_argument(
        "--events-out",
        default=(
            "results/"
            "citylearn_trust_aware_events.csv"
        ),
    )

    parser.add_argument(
        "--trace-out",
        default=(
            "results/"
            "citylearn_trust_aware_trace.csv"
        ),
    )

    args = parser.parse_args()

    (
        time_mean,
        safe_time_std,
        active_mask,
        ood_threshold,
    ) = load_trust_reference(
        args.reference,
        args.threshold,
    )

    print("=" * 80)
    print(
        "CITYLEARN TRUST-AWARE "
        "CONTROLLER EVALUATION"
    )
    print("=" * 80)

    print(
        f"Model:          {args.model}"
    )
    print(
        f"Reference:      {args.reference}"
    )
    print(
        f"OOD threshold:  "
        f"{args.threshold} = "
        f"{ood_threshold:.9f}"
    )
    print(
        f"Risk margin:    "
        f"{args.risk_margin:.3f}°C"
    )

    print(
        "\nLoading IQL controller..."
    )

    algo = load_iql_model(
        args.data,
        args.model,
    )

    print(
        "Controller loaded."
    )

    summary_rows = []
    event_rows = []
    trace_rows = []

    for (
        scenario,
        shift_c,
        schema,
    ) in SCENARIOS:

        if not Path(
            schema
        ).exists():
            raise FileNotFoundError(
                f"Missing scenario: {schema}"
            )

        print()
        print("#" * 80)
        print(
            f"SCENARIO {scenario} "
            f"({shift_c:+.1f}°C)"
        )
        print("#" * 80)

        for controller in CONTROLLERS:

            print()
            print("-" * 80)
            print(
                DISPLAY_NAMES[
                    controller
                ]
            )
            print("-" * 80)

            (
                metrics,
                events,
                trace,
            ) = evaluate_controller(
                algo=algo,
                schema=schema,
                scenario=scenario,
                shift_c=shift_c,
                controller=controller,
                max_steps=args.max_steps,
                time_mean=time_mean,
                safe_time_std=safe_time_std,
                active_mask=active_mask,
                ood_threshold=ood_threshold,
                threshold_name=args.threshold,
                risk_margin=args.risk_margin,
            )

            summary_rows.append(
                metrics
            )

            event_rows.extend(
                events
            )

            trace_rows.extend(
                trace
            )

            print_run_result(
                metrics
            )

    summary = pd.DataFrame(
        summary_rows
    )

    events = pd.DataFrame(
        event_rows
    )

    trace = pd.DataFrame(
        trace_rows
    )

    summary_path = Path(
        args.summary_out
    )

    events_path = Path(
        args.events_out
    )

    trace_path = Path(
        args.trace_out
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    events.to_csv(
        events_path,
        index=False,
    )

    trace.to_csv(
        trace_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("COMPACT COMPARISON")
    print("=" * 80)

    columns = [
        "scenario",
        "controller",
        "net_electricity_consumption",
        "worst_building_overheating_discomfort",
        "worst_building_overcooling_discomfort",
        "worst_building_combined_discomfort",
        "low_trust_fraction",
        "intervention_count",
    ]

    compact = summary[
        columns
    ].copy()

    print(
        compact.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print()
    print("Saved:")
    print(
        f"  {summary_path}"
    )
    print(
        f"  {events_path}"
    )
    print(
        f"  {trace_path}"
    )


if __name__ == "__main__":
    main()