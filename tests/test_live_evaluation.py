from pathlib import Path
import sys

# Add the project root to Python's import path so this test can be
# executed directly from the tests/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.citylearn_service import run_live_evaluation


def main():
    metrics, timeseries = run_live_evaluation("iql", 1)

    print("\n=== LIVE METRICS ===")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    district = timeseries[timeseries["level"] == "district"]

    print("\n=== TIMESERIES CHECK ===")
    print("Total rows:", len(timeseries))
    print("District steps:", len(district))
    print(
        "District mean net electricity:",
        district["net_electricity"].mean(),
    )

    print("\n=== FIRST ROWS ===")
    print(timeseries.head())


if __name__ == "__main__":
    main()