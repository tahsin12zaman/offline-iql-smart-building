import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = "results/citylearn_eval_scores.csv"
OUTPUT = "results/citylearn_3seed_comparison_polished.png"

df = pd.read_csv(INPUT)

# Convert negative reward to positive penalty magnitude.
# Lower penalty is better.
df["penalty"] = -df["score"]

summary = (
    df.groupby("algorithm")["penalty"]
    .agg(["mean", "std"])
    .reset_index()
)

print("\nCityLearn 3-seed summary:")
print(summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))

algorithms = ["BC", "IQL", "CQL"]

# Plot each individual seed.
for i, algorithm in enumerate(algorithms):
    values = df.loc[df["algorithm"] == algorithm, "penalty"].values

    x_positions = [
        i - 0.12,
        i,
        i + 0.12,
    ]

    ax.scatter(
        x_positions,
        values,
        s=70,
        zorder=3,
        label=None,
    )

# Plot mean ± standard deviation.
for i, algorithm in enumerate(algorithms):
    row = summary[summary["algorithm"] == algorithm].iloc[0]

    ax.errorbar(
        i,
        row["mean"],
        yerr=row["std"],
        fmt="o",
        markersize=9,
        capsize=6,
        linewidth=2,
        zorder=4,
    )

ax.set_yscale("log")

ax.set_xticks(range(len(algorithms)))
ax.set_xticklabels(algorithms)

ax.set_xlabel("Algorithm")
ax.set_ylabel("Negative evaluation score magnitude (lower is better)")
ax.set_title("CityLearn: 3-Seed Offline RL Evaluation")

ax.grid(
    axis="y",
    alpha=0.3,
    which="both",
)

plt.tight_layout()

Path("results").mkdir(exist_ok=True)
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved polished plot to: {OUTPUT}")