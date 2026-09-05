import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = "results/citylearn_eval_scores.csv"
OUTPUT = "results/citylearn_3seed_comparison.png"
SUMMARY_OUTPUT = "results/citylearn_3seed_summary.csv"

df = pd.read_csv(INPUT)

summary = (
    df.groupby("algorithm")["score"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

print("\nCityLearn 3-seed summary:")
print(summary.to_string(index=False))

Path("results").mkdir(exist_ok=True)
summary.to_csv(SUMMARY_OUTPUT, index=False)

fig, ax = plt.subplots(figsize=(9, 6))

algorithms = summary["algorithm"]
means = summary["mean"]
stds = summary["std"]

ax.bar(algorithms, means, yerr=stds, capsize=6)

ax.set_title("CityLearn: 3-Seed Offline RL Comparison")
ax.set_xlabel("Algorithm")
ax.set_ylabel("Evaluation Score")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=300)
plt.close()

print(f"\nSaved summary to: {SUMMARY_OUTPUT}")
print(f"Saved plot to: {OUTPUT}")