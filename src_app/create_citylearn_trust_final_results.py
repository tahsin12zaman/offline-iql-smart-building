from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"


MULTISEED_DELTAS = (
    RESULTS
    / "citylearn_trust_multiseed_delta_aggregate.csv"
)

TRUST_SIGNALS = (
    RESULTS
    / "citylearn_trust_signal_analysis.csv"
)

CONTROLLER_RESULTS = (
    RESULTS
    / "citylearn_trust_aware_summary.csv"
)

SENSITIVITY_RESULTS = (
    RESULTS
    / "citylearn_trust_sensitivity.csv"
)


FINAL_SUMMARY = (
    RESULTS
    / "citylearn_trust_final_summary.csv"
)

DETECTOR_SUMMARY = (
    RESULTS
    / "citylearn_trust_detector_summary.csv"
)

ABLATION_SUMMARY = (
    RESULTS
    / "citylearn_trust_controller_ablation.csv"
)

SELECTED_SENSITIVITY = (
    RESULTS
    / "citylearn_trust_selected_configuration.csv"
)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found: {path}"
        )


def scenario_order(df):
    order = {
        "Original": 0,
        "+1C": 1,
        "+2C": 2,
        "+3C": 3,
    }

    out = df.copy()

    out["_order"] = out[
        "scenario"
    ].map(order)

    out = (
        out
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    return out


def build_final_summary(multiseed):
    columns = [
        "scenario",
        "shift_c",
        "n_seeds",

        "combined_discomfort_reduction_pct_mean",
        "combined_discomfort_reduction_pct_std",
        "combined_discomfort_reduction_pct_ci95_halfwidth",
        "combined_discomfort_reduction_pct_min",
        "combined_discomfort_reduction_pct_max",

        "overheating_reduction_pct_mean",
        "overheating_reduction_pct_std",
        "overheating_reduction_pct_ci95_halfwidth",

        "energy_change_pct_mean",
        "energy_change_pct_std",
        "energy_change_pct_ci95_halfwidth",

        "cost_change_pct_mean",
        "cost_change_pct_std",
        "cost_change_pct_ci95_halfwidth",

        "carbon_change_pct_mean",
        "carbon_change_pct_std",
        "carbon_change_pct_ci95_halfwidth",

        "intervention_change_mean",
    ]

    missing = [
        c
        for c in columns
        if c not in multiseed.columns
    ]

    if missing:
        raise KeyError(
            f"Missing multiseed columns: {missing}"
        )

    result = multiseed[
        columns
    ].copy()

    result[
        "combined_discomfort_ci95_lower"
    ] = (
        result[
            "combined_discomfort_reduction_pct_mean"
        ]
        - result[
            "combined_discomfort_reduction_pct_ci95_halfwidth"
        ]
    )

    result[
        "combined_discomfort_ci95_upper"
    ] = (
        result[
            "combined_discomfort_reduction_pct_mean"
        ]
        + result[
            "combined_discomfort_reduction_pct_ci95_halfwidth"
        ]
    )

    result[
        "energy_change_ci95_lower"
    ] = (
        result[
            "energy_change_pct_mean"
        ]
        - result[
            "energy_change_pct_ci95_halfwidth"
        ]
    )

    result[
        "energy_change_ci95_upper"
    ] = (
        result[
            "energy_change_pct_mean"
        ]
        + result[
            "energy_change_pct_ci95_halfwidth"
        ]
    )

    return scenario_order(
        result
    )


def build_detector_summary(signals):
    columns = [
        "scenario",
        "shift_c",
        "mean_ood_score",
        "ood_p95_exceedance",
        "ood_p99_exceedance",
        "mean_policy_uncertainty",
        "policy_shift_ratio",
        "policy_p95_exceedance",
        "policy_p99_exceedance",
        "policy_auroc",
    ]

    result = signals[
        columns
    ].copy()

    return scenario_order(
        result
    )


def build_ablation_summary(controller_results):
    desired_controllers = [
        "IQL",
        "Comfort-Guarded IQL",
        "OOD-Gated IQL",
        "Trust-Aware IQL",
        "Trust-Aware + Guardrail",
    ]

    df = controller_results[
        controller_results[
            "controller"
        ].isin(
            desired_controllers
        )
    ].copy()

    columns = [
        "scenario",
        "shift_c",
        "controller",

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

    result = df[
        columns
    ].copy()

    scenario_rank = {
        "Original": 0,
        "+1C": 1,
        "+2C": 2,
        "+3C": 3,
    }

    controller_rank = {
        name: i
        for i, name
        in enumerate(
            desired_controllers
        )
    }

    result[
        "_scenario_order"
    ] = result[
        "scenario"
    ].map(
        scenario_rank
    )

    result[
        "_controller_order"
    ] = result[
        "controller"
    ].map(
        controller_rank
    )

    result = (
        result
        .sort_values(
            [
                "_scenario_order",
                "_controller_order",
            ]
        )
        .drop(
            columns=[
                "_scenario_order",
                "_controller_order",
            ]
        )
        .reset_index(drop=True)
    )

    return result


def build_selected_configuration(
    sensitivity,
):
    selected = sensitivity[
        (
            sensitivity[
                "threshold"
            ] == "p99"
        )
        &
        (
            np.isclose(
                sensitivity[
                    "risk_margin_c"
                ],
                0.50,
            )
        )
    ].copy()

    return scenario_order(
        selected
    )


def save_figure(fig, name):
    png = (
        FIGURES
        / f"{name}.png"
    )

    pdf = (
        FIGURES
        / f"{name}.pdf"
    )

    fig.tight_layout()

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_comfort_improvement(summary):
    fig, ax = plt.subplots(
        figsize=(7.5, 4.8)
    )

    x = summary[
        "shift_c"
    ].to_numpy()

    y = summary[
        "combined_discomfort_reduction_pct_mean"
    ].to_numpy()

    err = summary[
        "combined_discomfort_reduction_pct_ci95_halfwidth"
    ].to_numpy()

    ax.errorbar(
        x,
        y,
        yerr=err,
        marker="o",
        capsize=5,
        linewidth=2,
    )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.set_xticks(
        [0, 1, 2, 3]
    )

    ax.set_xticklabels(
        [
            "Original",
            "+1°C",
            "+2°C",
            "+3°C",
        ]
    )

    ax.set_xlabel(
        "Temperature distribution shift"
    )

    ax.set_ylabel(
        "Combined discomfort reduction vs\n"
        "Comfort-Guarded IQL (%)"
    )

    ax.set_title(
        "Multi-seed trust-aware comfort improvement"
    )

    ax.grid(
        alpha=0.25
    )

    save_figure(
        fig,
        "trust_multiseed_comfort_improvement",
    )


def plot_energy_cost(summary):
    fig, ax = plt.subplots(
        figsize=(7.5, 4.8)
    )

    x = summary[
        "shift_c"
    ].to_numpy()

    y = summary[
        "energy_change_pct_mean"
    ].to_numpy()

    err = summary[
        "energy_change_pct_ci95_halfwidth"
    ].to_numpy()

    ax.errorbar(
        x,
        y,
        yerr=err,
        marker="o",
        capsize=5,
        linewidth=2,
    )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.set_xticks(
        [0, 1, 2, 3]
    )

    ax.set_xticklabels(
        [
            "Original",
            "+1°C",
            "+2°C",
            "+3°C",
        ]
    )

    ax.set_xlabel(
        "Temperature distribution shift"
    )

    ax.set_ylabel(
        "Electricity change vs\n"
        "Comfort-Guarded IQL (%)"
    )

    ax.set_title(
        "Energy cost of trust-aware intervention"
    )

    ax.grid(
        alpha=0.25
    )

    save_figure(
        fig,
        "trust_multiseed_energy_overhead",
    )


def plot_tradeoff(summary):
    fig, ax = plt.subplots(
        figsize=(7.0, 5.2)
    )

    x = summary[
        "energy_change_pct_mean"
    ].to_numpy()

    y = summary[
        "combined_discomfort_reduction_pct_mean"
    ].to_numpy()

    xerr = summary[
        "energy_change_pct_ci95_halfwidth"
    ].to_numpy()

    yerr = summary[
        "combined_discomfort_reduction_pct_ci95_halfwidth"
    ].to_numpy()

    labels = summary[
        "scenario"
    ].tolist()

    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        capsize=4,
    )

    for xi, yi, label in zip(
        x,
        y,
        labels,
    ):
        ax.annotate(
            label,
            (xi, yi),
            xytext=(6, 5),
            textcoords="offset points",
        )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.axvline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Electricity increase vs "
        "Comfort-Guarded IQL (%)"
    )

    ax.set_ylabel(
        "Combined discomfort reduction vs "
        "Comfort-Guarded IQL (%)"
    )

    ax.set_title(
        "Trust-aware comfort–energy trade-off"
    )

    ax.grid(
        alpha=0.25
    )

    save_figure(
        fig,
        "trust_multiseed_comfort_energy_tradeoff",
    )


def plot_detector_severity(detector):
    fig, ax = plt.subplots(
        figsize=(7.5, 4.8)
    )

    x = detector[
        "shift_c"
    ].to_numpy()

    ood = detector[
        "mean_ood_score"
    ].to_numpy()

    ax.plot(
        x,
        ood,
        marker="o",
        linewidth=2,
    )

    ax.set_xticks(
        [0, 1, 2, 3]
    )

    ax.set_xticklabels(
        [
            "Original",
            "+1°C",
            "+2°C",
            "+3°C",
        ]
    )

    ax.set_xlabel(
        "Temperature distribution shift"
    )

    ax.set_ylabel(
        "Mean time-conditioned OOD score"
    )

    ax.set_title(
        "OOD signal tracks shift severity"
    )

    ax.grid(
        alpha=0.25
    )

    save_figure(
        fig,
        "trust_ood_shift_severity",
    )


def plot_policy_uncertainty(detector):
    fig, ax = plt.subplots(
        figsize=(7.5, 4.8)
    )

    shifted = detector[
        detector[
            "scenario"
        ] != "Original"
    ]

    x = shifted[
        "shift_c"
    ].to_numpy()

    auroc = shifted[
        "policy_auroc"
    ].to_numpy()

    ax.plot(
        x,
        auroc,
        marker="o",
        linewidth=2,
    )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xticks(
        [1, 2, 3]
    )

    ax.set_xticklabels(
        [
            "+1°C",
            "+2°C",
            "+3°C",
        ]
    )

    ax.set_ylim(
        0.45,
        max(
            0.60,
            float(
                np.nanmax(
                    auroc
                )
            )
            + 0.03,
        ),
    )

    ax.set_xlabel(
        "Temperature distribution shift"
    )

    ax.set_ylabel(
        "Policy-disagreement AUROC"
    )

    ax.set_title(
        "Ensemble disagreement is a weak "
        "per-state shift detector"
    )

    ax.grid(
        alpha=0.25
    )

    save_figure(
        fig,
        "trust_policy_uncertainty_auroc",
    )


def main():
    inputs = [
        MULTISEED_DELTAS,
        TRUST_SIGNALS,
        CONTROLLER_RESULTS,
        SENSITIVITY_RESULTS,
    ]

    for path in inputs:
        require_file(
            path
        )

    FIGURES.mkdir(
        parents=True,
        exist_ok=True,
    )

    multiseed = pd.read_csv(
        MULTISEED_DELTAS
    )

    trust_signals = pd.read_csv(
        TRUST_SIGNALS
    )

    controller_results = pd.read_csv(
        CONTROLLER_RESULTS
    )

    sensitivity = pd.read_csv(
        SENSITIVITY_RESULTS
    )

    final_summary = (
        build_final_summary(
            multiseed
        )
    )

    detector_summary = (
        build_detector_summary(
            trust_signals
        )
    )

    ablation = (
        build_ablation_summary(
            controller_results
        )
    )

    selected = (
        build_selected_configuration(
            sensitivity
        )
    )

    final_summary.to_csv(
        FINAL_SUMMARY,
        index=False,
    )

    detector_summary.to_csv(
        DETECTOR_SUMMARY,
        index=False,
    )

    ablation.to_csv(
        ABLATION_SUMMARY,
        index=False,
    )

    selected.to_csv(
        SELECTED_SENSITIVITY,
        index=False,
    )

    print("=" * 80)
    print("FINAL TRUST-AWARE MULTI-SEED SUMMARY")
    print("=" * 80)

    display_columns = [
        "scenario",
        "n_seeds",
        "combined_discomfort_reduction_pct_mean",
        "combined_discomfort_ci95_lower",
        "combined_discomfort_ci95_upper",
        "energy_change_pct_mean",
        "energy_change_ci95_lower",
        "energy_change_ci95_upper",
        "intervention_change_mean",
    ]

    print(
        final_summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print()
    print("=" * 80)
    print("DETECTOR SUMMARY")
    print("=" * 80)

    print(
        detector_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print()
    print("=" * 80)
    print("SELECTED CONFIGURATION")
    print("=" * 80)

    print(
        "OOD threshold: p99"
    )

    print(
        "Thermal-risk margin: 0.50°C"
    )

    print()
    print("Generating figures...")

    plot_comfort_improvement(
        final_summary
    )

    plot_energy_cost(
        final_summary
    )

    plot_tradeoff(
        final_summary
    )

    plot_detector_severity(
        detector_summary
    )

    plot_policy_uncertainty(
        detector_summary
    )

    print()
    print("Saved tables:")
    print(
        FINAL_SUMMARY
    )
    print(
        DETECTOR_SUMMARY
    )
    print(
        ABLATION_SUMMARY
    )
    print(
        SELECTED_SENSITIVITY
    )


if __name__ == "__main__":
    main()