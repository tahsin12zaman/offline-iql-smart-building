import pandas as pd

# Dataset variant summary
variant_df = pd.read_csv("results/dataset_variant_scores.csv")
variant_df = variant_df[variant_df["steps"] == 20000]
variant_df = variant_df.drop_duplicates(
    subset=["algorithm", "variant", "steps", "seed"],
    keep="last"
)

variant_summary = (
    variant_df.groupby(["variant", "algorithm"])["score"]
    .agg(["mean", "std"])
    .reset_index()
)

variant_summary["experiment"] = "dataset_variant"

# IQL quarter sweep summary
sweep_df = pd.read_csv("results/iql_quarter_sweep.csv")
sweep_df = sweep_df[sweep_df["steps"] == 20000]

sweep_summary = (
    sweep_df.groupby(["expectile", "weight_temp"])["score"]
    .agg(["mean", "std"])
    .reset_index()
)

sweep_summary["experiment"] = "iql_quarter_sweep"

# Low-data comparison
quarter_df = variant_df[variant_df["variant"] == "quarter"]

bc = quarter_df[quarter_df["algorithm"] == "BC"]["score"]
default_iql = quarter_df[quarter_df["algorithm"] == "IQL"]["score"]
cql = quarter_df[quarter_df["algorithm"] == "CQL"]["score"]

best_iql = sweep_summary.sort_values("mean", ascending=False).iloc[0]

low_data_rows = [
    {
        "method": "BC",
        "mean": bc.mean(),
        "std": bc.std(),
    },
    {
        "method": "Default IQL",
        "mean": default_iql.mean(),
        "std": default_iql.std(),
    },
    {
        "method": "Tuned IQL",
        "mean": best_iql["mean"],
        "std": best_iql["std"],
    },
    {
        "method": "CQL",
        "mean": cql.mean(),
        "std": cql.std(),
    },
]

low_data_summary = pd.DataFrame(low_data_rows)

variant_summary.to_csv("results/final_dataset_variant_summary.csv", index=False)
sweep_summary.to_csv("results/final_iql_quarter_sweep_summary.csv", index=False)
low_data_summary.to_csv("results/final_low_data_summary.csv", index=False)

print("Saved:")
print("results/final_dataset_variant_summary.csv")
print("results/final_iql_quarter_sweep_summary.csv")
print("results/final_low_data_summary.csv")
print()
print("Low-data summary:")
print(low_data_summary)