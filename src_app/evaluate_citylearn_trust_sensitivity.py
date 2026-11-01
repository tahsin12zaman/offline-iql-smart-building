from pathlib import Path
import argparse
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_APP = PROJECT_ROOT / "src_app"

if str(SRC_APP) not in sys.path:
    sys.path.insert(0, str(SRC_APP))


from evaluate_citylearn_guarded_v2 import load_iql_model

from evaluate_citylearn_trust_aware import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    DEFAULT_REFERENCE,
    SCENARIOS,
    load_trust_reference,
    evaluate_controller,
)


DEFAULT_OUT = (
    PROJECT_ROOT
    / "results"
    / "citylearn_trust_sensitivity.csv"
)


def percent_change(new_value, baseline):
    if baseline == 0:
        return float("nan")

    return (
        100.0
        * (new_value / baseline - 1.0)
    )


def percent_reduction(new_value, baseline):
    if baseline == 0:
        return float("nan")

    return (
        100.0
        * (baseline - new_value)
        / baseline
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sensitivity analysis for the "
            "CityLearn trust-aware controller."
        )
    )

    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
    )

    parser.add_argument(
        "--reference",
        default=str(DEFAULT_REFERENCE),
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        choices=[
            "p95",
            "p99",
        ],
        default=[
            "p95",
            "p99",
        ],
    )

    parser.add_argument(
        "--risk-margins",
        type=float,
        nargs="+",
        default=[
            0.0,
            0.25,
            0.50,
            0.75,
        ],
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=719,
    )

    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
    )

    args = parser.parse_args()

    print("=" * 80)
    print(
        "CITYLEARN TRUST-AWARE "
        "SENSITIVITY ANALYSIS"
    )
    print("=" * 80)

    print(
        f"Thresholds:  {args.thresholds}"
    )

    print(
        f"Risk margins:{args.risk_margins}"
    )

    print()
    print("Loading IQL controller...")

    algo = load_iql_model(
        args.data,
        args.model,
    )

    print("Controller loaded.")

    # ------------------------------------------------------------
    # Fixed comfort-guardrail baselines.
    # These do not depend on threshold or risk margin.
    # ------------------------------------------------------------

    (
        baseline_time_mean,
        baseline_safe_std,
        baseline_active_mask,
        baseline_threshold,
    ) = load_trust_reference(
        args.reference,
        "p95",
    )

    guarded_baselines = {}

    print()
    print("=" * 80)
    print("COMPUTING FIXED GUARDED BASELINES")
    print("=" * 80)

    for (
        scenario,
        shift_c,
        schema,
    ) in SCENARIOS:

        print(
            f"Guarded baseline: {scenario}"
        )

        (
            metrics,
            _,
            _,
        ) = evaluate_controller(
            algo=algo,
            schema=schema,
            scenario=scenario,
            shift_c=shift_c,
            controller="guarded",
            max_steps=args.max_steps,
            time_mean=baseline_time_mean,
            safe_time_std=baseline_safe_std,
            active_mask=baseline_active_mask,
            ood_threshold=baseline_threshold,
            threshold_name="p95",
            risk_margin=0.0,
        )

        guarded_baselines[
            scenario
        ] = metrics

    # ------------------------------------------------------------
    # Sensitivity sweep
    # ------------------------------------------------------------

    rows = []

    for threshold_name in args.thresholds:

        (
            time_mean,
            safe_time_std,
            active_mask,
            ood_threshold,
        ) = load_trust_reference(
            args.reference,
            threshold_name,
        )

        for risk_margin in args.risk_margins:

            print()
            print("#" * 80)
            print(
                f"THRESHOLD={threshold_name} "
                f"RISK_MARGIN="
                f"{risk_margin:.2f}°C"
            )
            print("#" * 80)

            for (
                scenario,
                shift_c,
                schema,
            ) in SCENARIOS:

                print(
                    f"Evaluating {scenario}..."
                )

                (
                    metrics,
                    _,
                    _,
                ) = evaluate_controller(
                    algo=algo,
                    schema=schema,
                    scenario=scenario,
                    shift_c=shift_c,
                    controller=(
                        "trust_aware_guarded"
                    ),
                    max_steps=args.max_steps,
                    time_mean=time_mean,
                    safe_time_std=safe_time_std,
                    active_mask=active_mask,
                    ood_threshold=ood_threshold,
                    threshold_name=threshold_name,
                    risk_margin=risk_margin,
                )

                baseline = (
                    guarded_baselines[
                        scenario
                    ]
                )

                combined = metrics[
                    "worst_building_combined_discomfort"
                ]

                baseline_combined = baseline[
                    "worst_building_combined_discomfort"
                ]

                energy = metrics[
                    "net_electricity_consumption"
                ]

                baseline_energy = baseline[
                    "net_electricity_consumption"
                ]

                rows.append({
                    "scenario":
                        scenario,

                    "shift_c":
                        shift_c,

                    "threshold":
                        threshold_name,

                    "ood_threshold":
                        ood_threshold,

                    "risk_margin_c":
                        risk_margin,

                    "reward":
                        metrics[
                            "reward"
                        ],

                    "net_electricity":
                        energy,

                    "electricity_cost":
                        metrics[
                            "electricity_cost"
                        ],

                    "carbon_emission":
                        metrics[
                            "carbon_emission"
                        ],

                    "peak_electricity":
                        metrics[
                            "peak_net_electricity"
                        ],

                    "worst_overheating":
                        metrics[
                            "worst_building_overheating_discomfort"
                        ],

                    "worst_overcooling":
                        metrics[
                            "worst_building_overcooling_discomfort"
                        ],

                    "worst_combined":
                        combined,

                    "low_trust_fraction":
                        metrics[
                            "low_trust_fraction"
                        ],

                    "intervention_count":
                        metrics[
                            "intervention_count"
                        ],

                    "guarded_worst_combined":
                        baseline_combined,

                    "guarded_energy":
                        baseline_energy,

                    "combined_reduction_vs_guarded_pct":
                        percent_reduction(
                            combined,
                            baseline_combined,
                        ),

                    "energy_change_vs_guarded_pct":
                        percent_change(
                            energy,
                            baseline_energy,
                        ),

                    "intervention_change_vs_guarded":
                        (
                            metrics[
                                "intervention_count"
                            ]
                            - baseline[
                                "intervention_count"
                            ]
                        ),
                })

    result = pd.DataFrame(
        rows
    )

    out_path = Path(
        args.out
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        out_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("SENSITIVITY SUMMARY")
    print("=" * 80)

    columns = [
        "scenario",
        "threshold",
        "risk_margin_c",
        "low_trust_fraction",
        "worst_combined",
        "combined_reduction_vs_guarded_pct",
        "net_electricity",
        "energy_change_vs_guarded_pct",
        "intervention_count",
    ]

    print(
        result[
            columns
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print()
    print(
        f"Saved: {out_path}"
    )


if __name__ == "__main__":
    main()