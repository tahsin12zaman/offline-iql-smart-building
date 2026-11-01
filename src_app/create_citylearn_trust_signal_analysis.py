from pathlib import Path
import argparse

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"

DEFAULT_OOD = (
    RESULTS
    / "citylearn_ood_severity_sweep.csv"
)

DEFAULT_POLICY_SCORES = (
    RESULTS
    / "citylearn_policy_uncertainty_scores.csv"
)

DEFAULT_POLICY_AUROC = (
    RESULTS
    / "citylearn_policy_uncertainty_auroc.csv"
)

DEFAULT_OUT = (
    RESULTS
    / "citylearn_trust_signal_analysis.csv"
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare CityLearn time-conditioned OOD "
            "scores with IQL ensemble policy uncertainty."
        )
    )

    parser.add_argument(
        "--ood",
        default=str(DEFAULT_OOD),
    )

    parser.add_argument(
        "--policy-scores",
        default=str(DEFAULT_POLICY_SCORES),
    )

    parser.add_argument(
        "--policy-auroc",
        default=str(DEFAULT_POLICY_AUROC),
    )

    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
    )

    args = parser.parse_args()

    ood = pd.read_csv(
        args.ood
    )

    policy = pd.read_csv(
        args.policy_scores
    )

    policy_auroc = pd.read_csv(
        args.policy_auroc
    )

    # ------------------------------------------------------------
    # Calibrate ensemble-uncertainty thresholds using ONLY
    # nominal/original-weather observations.
    # ------------------------------------------------------------

    original_policy = policy[
        policy["scenario"] == "Original"
    ]["policy_uncertainty"].to_numpy()

    if len(original_policy) == 0:
        raise ValueError(
            "No Original policy-uncertainty observations found."
        )

    policy_p95 = float(
        np.quantile(
            original_policy,
            0.95,
        )
    )

    policy_p99 = float(
        np.quantile(
            original_policy,
            0.99,
        )
    )

    print("=" * 80)
    print("CITYLEARN TRUST SIGNAL ANALYSIS")
    print("=" * 80)

    print()
    print("Nominal policy-uncertainty thresholds")
    print("-" * 80)
    print(
        f"P95: {policy_p95:.9f}"
    )
    print(
        f"P99: {policy_p99:.9f}"
    )

    scenario_to_shift = {
        "Original": 0,
        "+1C": 1,
        "+2C": 2,
        "+3C": 3,
    }

    rows = []

    original_policy_mean = float(
        np.mean(
            original_policy
        )
    )

    for scenario, shift_c in scenario_to_shift.items():

        policy_rows = policy[
            policy["scenario"] == scenario
        ]

        if len(policy_rows) == 0:
            raise ValueError(
                f"No policy uncertainty rows for {scenario}."
            )

        uncertainty = policy_rows[
            "policy_uncertainty"
        ].to_numpy()

        ood_rows = ood[
            ood["shift_c"] == shift_c
        ]

        if len(ood_rows) != 1:
            raise ValueError(
                f"Expected one OOD row for shift={shift_c}."
            )

        ood_row = ood_rows.iloc[0]

        mean_policy = float(
            np.mean(
                uncertainty
            )
        )

        policy_p95_exceedance = float(
            np.mean(
                uncertainty > policy_p95
            )
        )

        policy_p99_exceedance = float(
            np.mean(
                uncertainty > policy_p99
            )
        )

        if original_policy_mean > 0:
            policy_ratio = (
                mean_policy
                / original_policy_mean
            )
        else:
            policy_ratio = float(
                "nan"
            )

        if shift_c == 0:
            auroc = float(
                "nan"
            )
        else:
            match = policy_auroc[
                policy_auroc[
                    "scenario"
                ] == scenario
            ]

            if len(match) != 1:
                raise ValueError(
                    f"Missing AUROC for {scenario}."
                )

            auroc = float(
                match.iloc[0][
                    "auroc"
                ]
            )

        rows.append({
            "scenario": scenario,
            "shift_c": shift_c,

            "mean_ood_score": float(
                ood_row[
                    "mean_ood_score"
                ]
            ),

            "ood_p95_exceedance": float(
                ood_row[
                    "p95_exceedance"
                ]
            ),

            "ood_p99_exceedance": float(
                ood_row[
                    "p99_exceedance"
                ]
            ),

            "mean_policy_uncertainty":
                mean_policy,

            "policy_shift_ratio":
                policy_ratio,

            "policy_p95_exceedance":
                policy_p95_exceedance,

            "policy_p99_exceedance":
                policy_p99_exceedance,

            "policy_auroc":
                auroc,
        })

    result = pd.DataFrame(
        rows
    )

    # ------------------------------------------------------------
    # Aggregate severity trends
    # ------------------------------------------------------------

    ood_corr = result[
        "shift_c"
    ].corr(
        result[
            "mean_ood_score"
        ]
    )

    policy_corr = result[
        "shift_c"
    ].corr(
        result[
            "mean_policy_uncertainty"
        ]
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
    print("TRUST SIGNAL COMPARISON")
    print("=" * 80)

    print(
        result.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.9f}"
            ),
        )
    )

    print()
    print("SEVERITY CORRELATION")
    print("=" * 80)

    print(
        "Temperature shift vs mean OOD score: "
        f"{ood_corr:.6f}"
    )

    print(
        "Temperature shift vs mean policy "
        "uncertainty: "
        f"{policy_corr:.6f}"
    )

    print()
    print("INTERPRETATION")
    print("=" * 80)

    print(
        "Time-conditioned OOD is the primary "
        "distribution-shift signal."
    )

    print(
        "Ensemble disagreement is retained as "
        "an auxiliary confidence signal."
    )

    print(
        "Strong aggregate correlation alone does "
        "not imply strong per-state discrimination; "
        "AUROC must also be considered."
    )

    print()
    print(
        f"Saved: {out_path}"
    )


if __name__ == "__main__":
    main()