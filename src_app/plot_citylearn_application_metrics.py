import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


INPUT = "results/citylearn_application_metrics.csv"
SUMMARY_OUTPUT = "results/citylearn_application_metrics_summary.csv"

df = pd.read_csv(INPUT)

algorithm_order = ["BC", "IQL", "CQL"]

metrics = {
    "net_electricity_consumption": "Net Electricity Consumption",
    "electricity_cost": "Electricity Cost",
    "carbon_emission": "Carbon Emissions",
    "peak_net_electricity": "Peak Net Electricity",
}

summary_rows = []

for algorithm in algorithm_order:
    subset = df[df["algorithm"] == algorithm]

    row = {"algorithm": algorithm}

    for metric in metrics:
        row[f"{metric}_mean"] = subset[metric].mean()
        row[f"{metric}_std"] = subset[metric].std()

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

Path("results").mkdir(exist_ok=True)
summary.to_csv(SUMMARY_OUTPUT, index=False)

print("\nCityLearn Application Metric Summary")
print(summary.to_string(index=False))

for metric, title in metrics.items():

    means = []
    stds = []

    for algorithm in algorithm_order:
        subset = df[df["algorithm"] == algorithm]

        means.append(subset[metric].mean())
        stds.append(subset[metric].std())

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(
        algorithm_order,
        means,
        yerr=stds,
        capsize=6,
    )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel(title)
    ax.set_title(
        f"CityLearn Smart-Building Control: {title}\n"
        "Controlled 3-Seed Evaluation"
    )

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    filename = (
        "results/citylearn_"
        + metric
        + "_comparison.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved:", filename)

print("\nSaved summary:", SUMMARY_OUTPUT)