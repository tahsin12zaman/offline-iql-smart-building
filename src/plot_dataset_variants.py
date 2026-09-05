import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/dataset_variant_scores.csv")

# Keep only serious 20k-step runs.
df = df[df["steps"] == 20000]

# Remove accidental duplicate rows if any.
df = df.drop_duplicates(
    subset=["algorithm", "variant", "steps", "seed"],
    keep="last"
)

summary = (
    df.groupby(["variant", "algorithm"])["score"]
    .agg(["mean", "std"])
    .reset_index()
)

print(summary)

variants = ["full", "half", "quarter", "reward_noise", "action_noise"]
algorithms = ["BC", "IQL", "CQL"]

pivot = summary.pivot(index="variant", columns="algorithm", values="mean")
pivot = pivot.loc[variants, algorithms]

ax = pivot.plot(kind="bar", figsize=(10, 6))

plt.xlabel("Dataset Variant")
plt.ylabel("Average Evaluation Score")
plt.title("BC vs IQL vs CQL Under Dataset Quality Variants")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig("results/dataset_variant_comparison.png")
print("Saved plot to results/dataset_variant_comparison.png")