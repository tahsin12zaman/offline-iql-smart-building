from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CONTROLLERS = ["IQL", "Guarded IQL", "BasicRBC"]
WEATHERS = ["Original", "+3C"]


def load_iql(path, weather):
    df = pd.read_csv(path)
    rows = []

    for mode, controller in [
        ("NORMAL_IQL", "IQL"),
        ("GUARDED_IQL", "Guarded IQL"),
    ]:
        r = df.loc[df["mode"] == mode].iloc[0]

        rows.append({
            "weather": weather,
            "controller": controller,
            "reward": r["reward"],
            "net_electricity_consumption":
                r["net_electricity_consumption"],
            "electricity_cost": r["electricity_cost"],
            "carbon_emission": r["carbon_emission"],
            "peak_net_electricity": r["peak_net_electricity"],

            "b1_overheating":
                r["building_1_overheating_discomfort"],
            "b2_overheating":
                r["building_2_overheating_discomfort"],
            "b3_overheating":
                r["building_3_overheating_discomfort"],

            "b1_overcooling":
                r["building_1_overcooling_discomfort"],
            "b2_overcooling":
                r["building_2_overcooling_discomfort"],
            "b3_overcooling":
                r["building_3_overcooling_discomfort"],

            "b1_combined":
                r["building_1_combined_discomfort"],
            "b2_combined":
                r["building_2_combined_discomfort"],
            "b3_combined":
                r["building_3_combined_discomfort"],

            "worst_overheating":
                r["worst_building_overheating_discomfort"],
            "worst_overcooling":
                r["worst_building_overcooling_discomfort"],
            "worst_combined":
                r["worst_building_combined_discomfort"],

            "guardrail_activations":
                0 if mode == "NORMAL_IQL"
                else int(r["guardrail_activations"]),
        })

    return rows


def load_rbc(path, weather):
    r = pd.read_csv(path).iloc[0]

    return {
        "weather": weather,
        "controller": "BasicRBC",
        "reward": r["reward"],
        "net_electricity_consumption":
            r["net_electricity_consumption"],
        "electricity_cost": r["electricity_cost"],
        "carbon_emission": r["carbon_emission"],
        "peak_net_electricity": r["peak_net_electricity"],

        "b1_overheating":
            r["building_1_overheating_discomfort"],
        "b2_overheating":
            r["building_2_overheating_discomfort"],
        "b3_overheating":
            r["building_3_overheating_discomfort"],

        "b1_overcooling":
            r["building_1_overcooling_discomfort"],
        "b2_overcooling":
            r["building_2_overcooling_discomfort"],
        "b3_overcooling":
            r["building_3_overcooling_discomfort"],

        "b1_combined":
            r["building_1_combined_discomfort"],
        "b2_combined":
            r["building_2_combined_discomfort"],
        "b3_combined":
            r["building_3_combined_discomfort"],

        "worst_overheating":
            r["worst_building_overheating_discomfort"],
        "worst_overcooling":
            r["worst_building_overcooling_discomfort"],
        "worst_combined":
            r["worst_building_combined_discomfort"],

        "guardrail_activations": 0,
    }


def reduction(new, old):
    if old == 0:
        return np.nan
    return 100.0 * (old - new) / old


def change(new, old):
    if old == 0:
        return np.nan
    return 100.0 * (new - old) / old


# ------------------------------------------------------------------
# Load all six evaluations
# ------------------------------------------------------------------

rows = []

rows += load_iql(
    "results/original_weather_iql_comfort_summary.csv",
    "Original",
)

rows.append(
    load_rbc(
        "results/original_weather_rbc_summary.csv",
        "Original",
    )
)

rows += load_iql(
    "results/hot_weather_plus3C_iql_comfort_summary.csv",
    "+3C",
)

rows.append(
    load_rbc(
        "results/hot_weather_plus3C_rbc_summary.csv",
        "+3C",
    )
)

df = pd.DataFrame(rows)

df["weather"] = pd.Categorical(
    df["weather"],
    categories=WEATHERS,
    ordered=True,
)

df["controller"] = pd.Categorical(
    df["controller"],
    categories=CONTROLLERS,
    ordered=True,
)

df = df.sort_values(
    ["weather", "controller"]
).reset_index(drop=True)

df.to_csv(
    RESULTS_DIR / "citylearn_final_comparison.csv",
    index=False,
)


# ------------------------------------------------------------------
# Human-readable percentage table
# ------------------------------------------------------------------

display = df.copy()

comfort_cols = [
    "b1_overheating",
    "b2_overheating",
    "b3_overheating",
    "b1_overcooling",
    "b2_overcooling",
    "b3_overcooling",
    "b1_combined",
    "b2_combined",
    "b3_combined",
    "worst_overheating",
    "worst_overcooling",
    "worst_combined",
]

for col in comfort_cols:
    display[col] *= 100.0

display.to_csv(
    RESULTS_DIR / "citylearn_final_comparison_percent.csv",
    index=False,
)


# ------------------------------------------------------------------
# Key deltas
# ------------------------------------------------------------------

delta_rows = []

for weather in WEATHERS:
    d = df[df["weather"] == weather].set_index("controller")

    iql = d.loc["IQL"]
    guarded = d.loc["Guarded IQL"]
    rbc = d.loc["BasicRBC"]

    delta_rows.append({
        "weather": weather,

        "guarded_vs_iql_combined_discomfort_reduction_pct":
            reduction(
                guarded["worst_combined"],
                iql["worst_combined"],
            ),

        "guarded_vs_rbc_combined_discomfort_reduction_pct":
            reduction(
                guarded["worst_combined"],
                rbc["worst_combined"],
            ),

        "guarded_vs_iql_energy_change_pct":
            change(
                guarded["net_electricity_consumption"],
                iql["net_electricity_consumption"],
            ),

        "guarded_vs_rbc_energy_change_pct":
            change(
                guarded["net_electricity_consumption"],
                rbc["net_electricity_consumption"],
            ),

        "guarded_vs_iql_cost_change_pct":
            change(
                guarded["electricity_cost"],
                iql["electricity_cost"],
            ),

        "guarded_vs_rbc_cost_change_pct":
            change(
                guarded["electricity_cost"],
                rbc["electricity_cost"],
            ),

        "guarded_vs_iql_carbon_change_pct":
            change(
                guarded["carbon_emission"],
                iql["carbon_emission"],
            ),

        "guarded_vs_rbc_carbon_change_pct":
            change(
                guarded["carbon_emission"],
                rbc["carbon_emission"],
            ),

        "guardrail_activations":
            int(guarded["guardrail_activations"]),
    })

delta_df = pd.DataFrame(delta_rows)

delta_df.to_csv(
    RESULTS_DIR / "citylearn_final_deltas.csv",
    index=False,
)


# ------------------------------------------------------------------
# Figure 1: Worst combined discomfort
# ------------------------------------------------------------------

x = np.arange(len(WEATHERS))
width = 0.24

fig, ax = plt.subplots(figsize=(8.5, 5.2))

for i, controller in enumerate(CONTROLLERS):
    values = []

    for weather in WEATHERS:
        row = df[
            (df["weather"] == weather)
            & (df["controller"] == controller)
        ].iloc[0]

        values.append(100 * row["worst_combined"])

    bars = ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=controller,
    )

    ax.bar_label(
        bars,
        fmt="%.1f%%",
        padding=3,
        fontsize=9,
    )

ax.set_xticks(x)
ax.set_xticklabels([
    "Original Weather",
    "+3°C Weather",
])
ax.set_ylabel(
    "Worst-Building Combined Discomfort (%)"
)
ax.set_xlabel("Weather Scenario")
ax.set_title(
    "Thermal Comfort Comparison"
)
ax.legend()
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "combined_discomfort_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    FIGURES_DIR / "combined_discomfort_comparison.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ------------------------------------------------------------------
# Figure 2: Overheating vs overcooling
# ------------------------------------------------------------------

labels = []
overheating = []
overcooling = []

for weather in WEATHERS:
    for controller in CONTROLLERS:
        row = df[
            (df["weather"] == weather)
            & (df["controller"] == controller)
        ].iloc[0]

        labels.append(
            f"{controller}\n{weather}"
        )

        overheating.append(
            100 * row["worst_overheating"]
        )

        overcooling.append(
            100 * row["worst_overcooling"]
        )

x2 = np.arange(len(labels))
width2 = 0.36

fig, ax = plt.subplots(figsize=(11, 5.8))

bars1 = ax.bar(
    x2 - width2 / 2,
    overheating,
    width2,
    label="Overheating",
)

bars2 = ax.bar(
    x2 + width2 / 2,
    overcooling,
    width2,
    label="Overcooling",
)

ax.bar_label(
    bars1,
    fmt="%.1f%%",
    padding=2,
    fontsize=8,
)

ax.bar_label(
    bars2,
    fmt="%.1f%%",
    padding=2,
    fontsize=8,
)

ax.set_xticks(x2)
ax.set_xticklabels(labels)
ax.set_ylabel("Worst-Building Discomfort (%)")
ax.set_title(
    "Overheating and Overcooling Failure Modes"
)
ax.legend()
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "overheating_overcooling_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    FIGURES_DIR / "overheating_overcooling_comparison.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ------------------------------------------------------------------
# Figure 3: Energy vs combined comfort
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.5, 5.5))

markers = {
    "Original": "o",
    "+3C": "s",
}

for weather in WEATHERS:
    for controller in CONTROLLERS:
        row = df[
            (df["weather"] == weather)
            & (df["controller"] == controller)
        ].iloc[0]

        energy = row[
            "net_electricity_consumption"
        ]

        discomfort = (
            100 * row["worst_combined"]
        )

        ax.scatter(
            energy,
            discomfort,
            s=90,
            marker=markers[weather],
        )

        ax.annotate(
            f"{controller}\n{weather}",
            (energy, discomfort),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.5,
        )

ax.set_xlabel(
    "Net Electricity Consumption"
)
ax.set_ylabel(
    "Worst-Building Combined Discomfort (%)"
)
ax.set_title(
    "Energy–Comfort Trade-off"
)
ax.grid(alpha=0.25)
ax.set_axisbelow(True)

fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "energy_combined_comfort_tradeoff.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    FIGURES_DIR / "energy_combined_comfort_tradeoff.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ------------------------------------------------------------------
# Print final table
# ------------------------------------------------------------------

print("\n" + "=" * 100)
print("FINAL CITYLEARN RESULTS")
print("=" * 100)

for _, row in display.iterrows():
    print(
        f"{str(row['weather']):8s} | "
        f"{str(row['controller']):11s} | "
        f"Energy={row['net_electricity_consumption']:8.2f} | "
        f"Overheat={row['worst_overheating']:6.2f}% | "
        f"Overcool={row['worst_overcooling']:6.2f}% | "
        f"Combined={row['worst_combined']:6.2f}% | "
        f"Guard={int(row['guardrail_activations']):3d}"
    )

print("\n" + "=" * 100)
print("KEY DELTAS")
print("=" * 100)

print(
    delta_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

print("\nSaved final tables:")
print(
    "  results/citylearn_final_comparison.csv"
)
print(
    "  results/citylearn_final_comparison_percent.csv"
)
print(
    "  results/citylearn_final_deltas.csv"
)

print("\nSaved final figures:")
print(
    "  results/figures/combined_discomfort_comparison.png"
)
print(
    "  results/figures/overheating_overcooling_comparison.png"
)
print(
    "  results/figures/energy_combined_comfort_tradeoff.png"
)
