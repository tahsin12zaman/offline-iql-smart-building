import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/iql_quarter_sweep.csv")

summary = (
    df.groupby(["expectile", "weight_temp"])["score"]
    .agg(["mean", "std"])
    .reset_index()
)

print(summary)

# Create readable labels
summary["setting"] = (
    "exp="
    + summary["expectile"].astype(str)
    + ", temp="
    + summary["weight_temp"].astype(str)
)

summary = summary.sort_values("mean", ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(summary["setting"], summary["mean"], yerr=summary["std"], capsize=5)
plt.xlabel("IQL Hyperparameter Setting")
plt.ylabel("Average Evaluation Score")
plt.title("IQL Hyperparameter Sweep on Quarter Dataset")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("results/iql_quarter_sweep.png")
print("Saved plot to results/iql_quarter_sweep.png")

best = summary.iloc[0]
print()
print("Best setting:")
print("Expectile:", best["expectile"])
print("Weight temp:", best["weight_temp"])
print("Mean score:", best["mean"])
print("Std:", best["std"])