from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import t


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_APP = PROJECT_ROOT / "src_app"

if str(SRC_APP) not in sys.path:
    sys.path.insert(0, str(SRC_APP))


from evaluate_citylearn_guarded_v2 import (
    load_iql_model,
)

from evaluate_citylearn_trust_aware import (
    DEFAULT_DATA,
    DEFAULT_REFERENCE,
    SCENARIOS,
    load_trust_reference,
    evaluate_controller,
)


DEFAULT_ENSEMBLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "trust_ensemble"
)

DEFAULT_RUNS_OUT = (
    PROJECT_ROOT
    / "results"
    / "citylearn_trust_multiseed_runs.csv"
)

DEFAULT_AGGREGATE_OUT = (
    PROJECT_ROOT
    / "results"
    / "citylearn_trust_multiseed_aggregate.csv"
)

DEFAULT_DELTAS_OUT = (
    PROJECT_ROOT
    / "results"
    / "citylearn_trust_multiseed_deltas.csv"
)

DEFAULT_DELTA_AGGREGATE_OUT = (
    PROJECT_ROOT
    / "results"
    / "citylearn_trust_multiseed_delta_aggregate.csv"
)


CONTROLLERS = [
    "iql",
    "guarded",
    "trust_aware_guarded",
]


DISPLAY_NAMES = {
    "iql": "IQL",
    "guarded": "Comfort-Guarded IQL",
    "trust_aware_guarded":
        "Trust-Aware + Guardrail",
}


def percent_change(
    new_value,
    baseline_value,
):
    if baseline_value == 0:
        return float("nan")

    return (
        100.0
        * (
            new_value
            / baseline_value
            - 1.0
        )
    )


def percent_reduction(
    new_value,
    baseline_value,
):
    if baseline_value == 0:
        return float("nan")

    return (
        100.0
        * (
            baseline_value
            - new_value
        )
        / baseline_value
    )


def ci95_halfwidth(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n < 2:
        return float("nan")

    std = np.std(
        values,
        ddof=1,
    )

    sem = std / np.sqrt(n)

    critical = t.ppf(
        0.975,
        df=n - 1,
    )

    return float(
        critical * sem
    )


def aggregate_runs(df):
    metric_columns = [
        "reward",
        "net_electricity_consumption",
        "electricity_cost",
        "carbon_emission",
        "peak_net_electricity",
        "worst_building_overheating_discomfort",
        "worst_building_overcooling_discomfort",
        "worst_building_combined_discomfort",
        "low_trust_fraction",
        "intervention_count",
    ]

    rows = []

    grouped = df.groupby(
        [
            "scenario",
            "shift_c",
            "controller",
            "controller_key",
        ],
        sort=False,
    )

    for keys, group in grouped:
        (
            scenario,
            shift_c,
            controller,
            controller_key,
        ) = keys

        row = {
            "scenario": scenario,
            "shift_c": shift_c,
            "controller": controller,
            "controller_key": controller_key,
            "n_seeds": int(
                group["seed"].nunique()
            ),
        }

        for metric in metric_columns:
            values = group[
                metric
            ].to_numpy(
                dtype=float
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.mean(values)
            )

            row[
                f"{metric}_std"
            ] = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

            row[
                f"{metric}_ci95_halfwidth"
            ] = ci95_halfwidth(
                values
            )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def build_seed_deltas(runs):
    rows = []

    grouped = runs.groupby(
        [
            "seed",
            "scenario",
            "shift_c",
        ],
        sort=False,
    )

    for (
        seed,
        scenario,
        shift_c,
    ), group in grouped:

        guarded_rows = group[
            group[
                "controller_key"
            ] == "guarded"
        ]

        trust_rows = group[
            group[
                "controller_key"
            ] == "trust_aware_guarded"
        ]

        if (
            len(guarded_rows) != 1
            or len(trust_rows) != 1
        ):
            raise RuntimeError(
                "Expected one guarded and one "
                "trust-aware row per "
                f"seed/scenario. seed={seed}, "
                f"scenario={scenario}"
            )

        guarded = guarded_rows.iloc[0]
        trust = trust_rows.iloc[0]

        rows.append({
            "seed": seed,
            "scenario": scenario,
            "shift_c": shift_c,

            "guarded_combined_discomfort":
                guarded[
                    "worst_building_combined_discomfort"
                ],

            "trust_combined_discomfort":
                trust[
                    "worst_building_combined_discomfort"
                ],

            "combined_discomfort_reduction_pct":
                percent_reduction(
                    trust[
                        "worst_building_combined_discomfort"
                    ],
                    guarded[
                        "worst_building_combined_discomfort"
                    ],
                ),

            "guarded_overheating":
                guarded[
                    "worst_building_overheating_discomfort"
                ],

            "trust_overheating":
                trust[
                    "worst_building_overheating_discomfort"
                ],

            "overheating_reduction_pct":
                percent_reduction(
                    trust[
                        "worst_building_overheating_discomfort"
                    ],
                    guarded[
                        "worst_building_overheating_discomfort"
                    ],
                ),

            "guarded_energy":
                guarded[
                    "net_electricity_consumption"
                ],

            "trust_energy":
                trust[
                    "net_electricity_consumption"
                ],

            "energy_change_pct":
                percent_change(
                    trust[
                        "net_electricity_consumption"
                    ],
                    guarded[
                        "net_electricity_consumption"
                    ],
                ),

            "guarded_cost":
                guarded[
                    "electricity_cost"
                ],

            "trust_cost":
                trust[
                    "electricity_cost"
                ],

            "cost_change_pct":
                percent_change(
                    trust[
                        "electricity_cost"
                    ],
                    guarded[
                        "electricity_cost"
                    ],
                ),

            "guarded_carbon":
                guarded[
                    "carbon_emission"
                ],

            "trust_carbon":
                trust[
                    "carbon_emission"
                ],

            "carbon_change_pct":
                percent_change(
                    trust[
                        "carbon_emission"
                    ],
                    guarded[
                        "carbon_emission"
                    ],
                ),

            "guarded_interventions":
                guarded[
                    "intervention_count"
                ],

            "trust_interventions":
                trust[
                    "intervention_count"
                ],

            "intervention_change":
                (
                    trust[
                        "intervention_count"
                    ]
                    - guarded[
                        "intervention_count"
                    ]
                ),
        })

    return pd.DataFrame(
        rows
    )


def aggregate_deltas(deltas):
    metric_columns = [
        "combined_discomfort_reduction_pct",
        "overheating_reduction_pct",
        "energy_change_pct",
        "cost_change_pct",
        "carbon_change_pct",
        "intervention_change",
    ]

    rows = []

    grouped = deltas.groupby(
        [
            "scenario",
            "shift_c",
        ],
        sort=False,
    )

    for (
        scenario,
        shift_c,
    ), group in grouped:

        row = {
            "scenario": scenario,
            "shift_c": shift_c,
            "n_seeds": int(
                group[
                    "seed"
                ].nunique()
            ),
        }

        for metric in metric_columns:
            values = group[
                metric
            ].to_numpy(
                dtype=float
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.mean(values)
            )

            row[
                f"{metric}_std"
            ] = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

            row[
                f"{metric}_ci95_halfwidth"
            ] = ci95_halfwidth(
                values
            )

            row[
                f"{metric}_min"
            ] = float(
                np.min(values)
            )

            row[
                f"{metric}_max"
            ] = float(
                np.max(values)
            )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-seed robustness evaluation "
            "for CityLearn trust-aware IQL."
        )
    )

    parser.add_argument(
        "--data",
        default=str(
            DEFAULT_DATA
        ),
    )

    parser.add_argument(
        "--ensemble-dir",
        default=str(
            DEFAULT_ENSEMBLE_DIR
        ),
    )

    parser.add_argument(
        "--reference",
        default=str(
            DEFAULT_REFERENCE
        ),
    )

    parser.add_argument(
        "--threshold",
        choices=[
            "p95",
            "p99",
        ],
        default="p99",
    )

    parser.add_argument(
        "--risk-margin",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=719,
    )

    parser.add_argument(
        "--runs-out",
        default=str(
            DEFAULT_RUNS_OUT
        ),
    )

    parser.add_argument(
        "--aggregate-out",
        default=str(
            DEFAULT_AGGREGATE_OUT
        ),
    )

    parser.add_argument(
        "--deltas-out",
        default=str(
            DEFAULT_DELTAS_OUT
        ),
    )

    parser.add_argument(
        "--delta-aggregate-out",
        default=str(
            DEFAULT_DELTA_AGGREGATE_OUT
        ),
    )

    args = parser.parse_args()

    ensemble_dir = Path(
        args.ensemble_dir
    )

    model_paths = sorted(
        ensemble_dir.glob(
            "citylearn_iql_seed_*.d3"
        )
    )

    if len(model_paths) != 5:
        raise RuntimeError(
            "Expected exactly 5 reproducible "
            "IQL models. "
            f"Found {len(model_paths)}: "
            f"{model_paths}"
        )

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
        "MULTI-SEED EVALUATION"
    )
    print("=" * 80)

    print(
        f"Threshold:   "
        f"{args.threshold} "
        f"({ood_threshold:.9f})"
    )

    print(
        f"Risk margin: "
        f"{args.risk_margin:.2f}°C"
    )

    print(
        f"Models:      {len(model_paths)}"
    )

    run_rows = []

    for model_path in model_paths:

        seed_text = (
            model_path.stem
            .split("_seed_")[-1]
        )

        seed = int(
            seed_text
        )

        print()
        print("#" * 80)
        print(
            f"SEED {seed}"
        )
        print("#" * 80)

        algo = load_iql_model(
            args.data,
            model_path,
        )

        for (
            scenario,
            shift_c,
            schema,
        ) in SCENARIOS:

            print()
            print(
                f"{scenario} "
                f"({shift_c:+.1f}°C)"
            )

            for controller in CONTROLLERS:

                print(
                    "  "
                    + DISPLAY_NAMES[
                        controller
                    ]
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
                    controller=controller,
                    max_steps=args.max_steps,
                    time_mean=time_mean,
                    safe_time_std=safe_time_std,
                    active_mask=active_mask,
                    ood_threshold=ood_threshold,
                    threshold_name=args.threshold,
                    risk_margin=args.risk_margin,
                )

                metrics[
                    "seed"
                ] = seed

                run_rows.append(
                    metrics
                )

    runs = pd.DataFrame(
        run_rows
    )

    aggregate = aggregate_runs(
        runs
    )

    deltas = build_seed_deltas(
        runs
    )

    delta_aggregate = (
        aggregate_deltas(
            deltas
        )
    )

    runs_path = Path(
        args.runs_out
    )

    aggregate_path = Path(
        args.aggregate_out
    )

    deltas_path = Path(
        args.deltas_out
    )

    delta_aggregate_path = Path(
        args.delta_aggregate_out
    )

    runs_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs.to_csv(
        runs_path,
        index=False,
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    deltas.to_csv(
        deltas_path,
        index=False,
    )

    delta_aggregate.to_csv(
        delta_aggregate_path,
        index=False,
    )

    print()
    print("=" * 80)
    print(
        "TRUST-AWARE VS FIXED-GUARDRAIL "
        "MULTI-SEED DELTAS"
    )
    print("=" * 80)

    columns = [
        "scenario",
        "n_seeds",

        "combined_discomfort_reduction_pct_mean",
        "combined_discomfort_reduction_pct_std",
        "combined_discomfort_reduction_pct_ci95_halfwidth",

        "energy_change_pct_mean",
        "energy_change_pct_std",
        "energy_change_pct_ci95_halfwidth",

        "intervention_change_mean",
    ]

    print(
        delta_aggregate[
            columns
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(
        f"Runs:            {runs_path}"
    )

    print(
        f"Aggregate:       {aggregate_path}"
    )

    print(
        f"Seed deltas:     {deltas_path}"
    )

    print(
        "Delta aggregate: "
        f"{delta_aggregate_path}"
    )


if __name__ == "__main__":
    main()