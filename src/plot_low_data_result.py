import pandas as pd
import matplotlib.pyplot as plt

# Load dataset-variant baseline results
variant_df = pd.read_csv("results/dataset_variant_scores.csv")
variant_df = variant_df[variant_df["steps"] == 20000]

# Remove duplicate accidental rows if any
variant_df = variant_df.drop_duplicates(
    subset=["algorithm", "variant", "steps", "seed"],
    keep="last"
)

# Quarter baseline results
quarter_df = variant_df[variant_df["variant"] == "quarter"]

bc_mean = quarter_df[quarter_df["algorithm"] == "BC"]["score"].mean()
cql_mean = quarter_df[quarter_df["algorithm"] == "CQL"]["score"].mean()
default_iql_mean = quarter_df[quarter_df["algorithm"] == "IQL"]["score"].mean()

bc_std = quarter_df[quarter_df["algorithm"] == "BC"]["score"].std()
cql_std = quarter_df[quarter_df["algorithm"] == "CQL"]["score"].std()
default_iql_std = quarter_df[quarter_df["algorithm"] == "IQL"]["score"].std()

# Load IQL sweep results
sweep_df = pd.read_csv("results/iql_quarter_sweep.csv")
sweep_df = sweep_df[sweep_df["steps"] == 20000]

summary = (
    sweep_df.groupby(["expectile", "weight_temp"])["score"]
    .agg(["mean", "std"])
    .reset_index()
)

best = summary.sort_values("mean", ascending=False).iloc[0]

tuned_iql_mean = best["mean"]
tuned_iql_std = best["std"]
best_expectile = best["expectile"]
best_weight_temp = best["weight_temp"]

print("Quarter dataset comparison")
print("==========================")
print("BC mean:", bc_mean, "std:", bc_std)
print("Default IQL mean:", default_iql_mean, "std:", default_iql_std)
print("Tuned IQL mean:", tuned_iql_mean, "std:", tuned_iql_std)
print("CQL mean:", cql_mean, "std:", cql_std)
print()
print("Best tuned IQL setting:")
print("expectile:", best_expectile)
print("weight_temp:", best_weight_temp)

labels = [
    "BC",
    "Default IQL\nexp=0.7,temp=3.0",
    "Tuned IQL\nexp=0.5,temp=10.0",
    "CQL",
]

means = [
    bc_mean,
    default_iql_mean,
    tuned_iql_mean,
    cql_mean,
]

stds = [
    bc_std,
    default_iql_std,
    tuned_iql_std,
    cql_std,
]

plt.figure(figsize=(9, 6))
plt.bar(labels, means, yerr=stds, capsize=5)

plt.ylabel("Average Evaluation Score")
plt.title("Low-Data Quarter Dataset: Default vs Tuned IQL")
plt.tight_layout()

plt.savefig("results/low_data_iql_improvement.png")
print("Saved plot to results/low_data_iql_improvement.png")