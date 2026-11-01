from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    PROJECT_ROOT
    / "external"
    / "CityLearn"
    / "data"
    / "datasets"
    / "citylearn_challenge_2023_phase_2_local_evaluation"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "scenarios"
    / "hot_weather_plus3C"
)

TEMPERATURE_COLUMNS = [
    "outdoor_dry_bulb_temperature",
    "outdoor_dry_bulb_temperature_predicted_1",
    "outdoor_dry_bulb_temperature_predicted_2",
    "outdoor_dry_bulb_temperature_predicted_3",
]

TEMPERATURE_SHIFT_C = 3.0


def main():
    parser = argparse.ArgumentParser(
        description="Create a CityLearn scenario with outdoor temperatures increased by 3°C."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output scenario if it already exists.",
    )
    args = parser.parse_args()

    source_schema = SOURCE_DIR / "schema.json"
    source_weather = SOURCE_DIR / "weather.csv"

    if not source_schema.is_file():
        raise FileNotFoundError(
            f"Source schema not found: {source_schema}"
        )

    if not source_weather.is_file():
        raise FileNotFoundError(
            f"Source weather file not found: {source_weather}"
        )

    if OUTPUT_DIR.exists():
        if not args.force:
            raise FileExistsError(
                f"Output already exists: {OUTPUT_DIR}\n"
                "Use --force to recreate it."
            )
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Copy the complete dataset so the original remains unchanged.
    shutil.copytree(SOURCE_DIR, OUTPUT_DIR)

    weather_path = OUTPUT_DIR / "weather.csv"
    weather = pd.read_csv(weather_path)

    missing_columns = [
        column
        for column in TEMPERATURE_COLUMNS
        if column not in weather.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing expected weather columns: "
            + ", ".join(missing_columns)
        )

    # Apply the same temperature shift to actual weather and forecasts.
    for column in TEMPERATURE_COLUMNS:
        weather[column] = (
            pd.to_numeric(weather[column], errors="raise")
            + TEMPERATURE_SHIFT_C
        )

    weather.to_csv(weather_path, index=False)

    # Point CityLearn to the copied scenario's files.
    schema_path = OUTPUT_DIR / "schema.json"

    with schema_path.open("r", encoding="utf-8") as file:
        schema = json.load(file)

    schema["root_directory"] = str(OUTPUT_DIR.resolve())

    with schema_path.open("w", encoding="utf-8") as file:
        json.dump(schema, file, indent=2)
        file.write("\n")

    print("Scenario created successfully.")
    print(f"Scenario directory: {OUTPUT_DIR}")
    print(f"Scenario schema:    {schema_path}")
    print(f"Weather file:       {weather_path}")
    print(f"Temperature shift:  +{TEMPERATURE_SHIFT_C}°C")
    print(f"Weather rows:       {len(weather)}")
    print("\nUpdated temperature columns:")
    for column in TEMPERATURE_COLUMNS:
        print(f"  {column}")


if __name__ == "__main__":
    main()