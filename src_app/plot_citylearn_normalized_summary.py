import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


INPUT = "results/citylearn_application_metrics_summary.csv"
OUTPUT = "results/citylearn_normalized_application_summary.png"

df = pd.read_csv(INPUT)

algorithm_order = ["BC", "IQL", "CQL"]

metrics = {
    "net_electricity_consumption_mean": "Electricity\nConsumption",
    "electricity_cost_mean": "Electricity\nCost",
    "carbon_emission_mean": "Carbon\nEmissions",
    "peak_net_electricity_mean": "Peak\nDemand",
}

df = df.set_index("algorithm").loc[algorithm_order]

# Normalize every metric relative to BC.
# BC = 100%. Lower is better.
normalized = pd.DataFrame(index=algorithm_order)

for column, label in metrics.items():
    bc_value = df.loc["BC", column]

    normalized[label] = (
        df[column] / bc_value
    ) * 100.0


print("\nNormalized CityLearn Application Metrics")
print("BC = 100%; lower is better\n")
print(normalized.to_string())


fig, ax = plt.subplots(figsize=(10, 6))

normalized.T.plot(
    kind="bar",
    ax=ax,
)

ax.axhline(
    100,
    linestyle="--",
    linewidth=1.5,
)

ax.set_ylabel("Performance Relative to BC (%)")
ax.set_xlabel("Application Metric")

ax.set_title(
    "CityLearn Smart-Building Control:\n"
    "Application Performance Relative to Behavior Cloning"
)

ax.legend(
    title="Algorithm",
)

ax.grid(
    axis="y",
    alpha=0.3,
)

plt.xticks(
    rotation=0,
)

plt.tight_layout()

Path("results").mkdir(exist_ok=True)

plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nSaved:", OUTPUT)