import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/scores.csv")

# Use only the serious baseline experiment, not the early 5000-step sanity checks
df = df[df["steps"] == 20000]

summary = df.groupby("algorithm")["score"].agg(["mean", "std"]).reset_index()

print(summary)

plt.figure(figsize=(8, 5))
plt.bar(summary["algorithm"], summary["mean"], yerr=summary["std"], capsize=5)
plt.xlabel("Algorithm")
plt.ylabel("Evaluation Score")
plt.title("Offline RL Baseline Comparison on Pendulum - 20k Steps")
plt.tight_layout()

plt.savefig("results/baseline_comparison_20k.png")
print("Saved plot to results/baseline_comparison_20k.png")