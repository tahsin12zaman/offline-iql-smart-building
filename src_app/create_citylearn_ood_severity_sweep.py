from pathlib import Path
import argparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"

DEFAULT_PLUS1 = (
    RESULTS
    / "citylearn_time_ood_plus1C_summary.csv"
)

DEFAULT_PLUS2 = (
    RESULTS
    / "citylearn_time_ood_plus2C_summary.csv"
)

DEFAULT_PLUS3 = (
    RESULTS
    / "citylearn_time_ood_summary.csv"
)

DEFAULT_OUT = (
    RESULTS
    / "citylearn_ood_severity_sweep.csv"
)


def get_original_row(df):
    rows = df[
        df["scenario"] == "Original"
    ]

    if len(rows) != 1:
        raise ValueError(
            "Expected exactly one Original row."
        )

    return rows.iloc[0]


def get_shifted_row(df):
    rows = df[
        df["scenario"] != "Original"
    ]

    if len(rows) != 1:
        raise ValueError(
            "Expected exactly one shifted row."
        )

    return rows.iloc[0]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine CityLearn time-conditioned "
            "OOD results into a temperature-shift "
            "severity sweep."
        )
    )

    parser.add_argument(
        "--plus1",
        default=str(DEFAULT_PLUS1),
    )

    parser.add_argument(
        "--plus2",
        default=str(DEFAULT_PLUS2),
    )

    parser.add_argument(
        "--plus3",
        default=str(DEFAULT_PLUS3),
    )

    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
    )

    args = parser.parse_args()

    inputs = {
        1: Path(args.plus1),
        2: Path(args.plus2),
        3: Path(args.plus3),
    }

    for shift, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing +{shift}C summary: {path}"
            )

    plus1 = pd.read_csv(
        inputs[1]
    )

    original = get_original_row(
        plus1
    )

    rows = [{
        "shift_c": 0,
        "scenario": "Original",
        "mean_ood_score": float(
            original["mean_score"]
        ),
        "median_ood_score": float(
            original["median_score"]
        ),
        "p95_ood_score": float(
            original["p95_score"]
        ),
        "p99_ood_score": float(
            original["p99_score"]
        ),
        "p95_exceedance": float(
            original[
                "fraction_above_threshold_95"
            ]
        ),
        "p99_exceedance": float(
            original[
                "fraction_above_threshold_99"
            ]
        ),
    }]

    for shift_c, path in inputs.items():
        df = pd.read_csv(
            path
        )

        shifted = get_shifted_row(
            df
        )

        rows.append({
            "shift_c": shift_c,
            "scenario": f"+{shift_c}C",
            "mean_ood_score": float(
                shifted["mean_score"]
            ),
            "median_ood_score": float(
                shifted["median_score"]
            ),
            "p95_ood_score": float(
                shifted["p95_score"]
            ),
            "p99_ood_score": float(
                shifted["p99_score"]
            ),
            "p95_exceedance": float(
                shifted[
                    "fraction_above_threshold_95"
                ]
            ),
            "p99_exceedance": float(
                shifted[
                    "fraction_above_threshold_99"
                ]
            ),
        })

    result = (
        pd.DataFrame(rows)
        .sort_values("shift_c")
        .reset_index(drop=True)
    )

    # Quantify whether continuous OOD score
    # increases with shift magnitude.
    severity_correlation = result[
        "shift_c"
    ].corr(
        result[
            "mean_ood_score"
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

    print("=" * 80)
    print("CITYLEARN OOD SEVERITY SWEEP")
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
    print(
        "Correlation between temperature "
        "shift and mean OOD score: "
        f"{severity_correlation:.6f}"
    )

    print()
    print(
        f"Saved: {out_path}"
    )


if __name__ == "__main__":
    main()