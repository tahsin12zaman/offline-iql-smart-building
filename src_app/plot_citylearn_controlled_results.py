import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = "results/citylearn_eval_scores.csv"
SUMMARY_OUTPUT = "results/citylearn_controlled_3seed_summary.csv"
PLOT_OUTPUT = "results/citylearn_controlled_3seed_comparison.png"

df = pd.read_csv(INPUT)

# Final controlled experiment uses only properly seeded runs.
df = df[df["seed"].isin([1, 2, 3])].copy()

algorithm_order = ["BC", "IQL", "CQL"]

summary = (
    df.groupby("algorithm")["score"]
    .agg(["mean", "std", "count"])
    .reindex(algorithm_order)
    .reset_index()
)

Path("results").mkdir(exist_ok=True)
summary.to_csv(SUMMARY_OUTPUT, index=False)

print("\nControlled CityLearn Results — Seeds 1, 2, 3")
print(summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 6))

for i, algorithm in enumerate(algorithm_order):
    values = df.loc[df["algorithm"] == algorithm, "score"].values

    offsets = [-0.08, 0.0, 0.08]

    for offset, value in zip(offsets, values):
        ax.scatter(
            i + offset,
            value,
            s=70,
            zorder=3,
        )

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

ax.set_xticks(range(len(algorithm_order)))
ax.set_xticklabels(algorithm_order)

ax.set_xlabel("Algorithm")
ax.set_ylabel("Cumulative CityLearn Reward (higher is better)")
ax.set_title("CityLearn Smart-Building Control: Controlled 3-Seed Evaluation")

ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_OUTPUT, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved summary:", SUMMARY_OUTPUT)
print("Saved plot:", PLOT_OUTPUT)