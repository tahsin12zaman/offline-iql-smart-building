from pathlib import Path
import argparse
import json
import shutil

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    PROJECT_ROOT
    / "external"
    / "CityLearn"
    / "data"
    / "datasets"
    / "citylearn_challenge_2023_phase_2_local_evaluation"
)

TEMPERATURE_COLUMNS = [
    "outdoor_dry_bulb_temperature",
    "outdoor_dry_bulb_temperature_predicted_1",
    "outdoor_dry_bulb_temperature_predicted_2",
    "outdoor_dry_bulb_temperature_predicted_3",
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--delta",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--name",
        required=True,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    output = (
        PROJECT_ROOT
        / "data"
        / "scenarios"
        / args.name
    )

    if output.exists():
        if not args.force:
            raise FileExistsError(
                f"{output} already exists. Use --force."
            )

        shutil.rmtree(output)

    shutil.copytree(
        SOURCE,
        output,
    )

    weather_path = output / "weather.csv"

    weather = pd.read_csv(
        weather_path
    )

    for column in TEMPERATURE_COLUMNS:
        if column not in weather.columns:
            raise KeyError(
                f"Missing weather column: {column}"
            )

        weather[column] = (
            weather[column]
            + args.delta
        )

    weather.to_csv(
        weather_path,
        index=False,
    )

    schema_path = output / "schema.json"

    schema = json.loads(
        schema_path.read_text()
    )

    schema["root_directory"] = str(
        output.resolve()
    )

    schema_path.write_text(
        json.dumps(
            schema,
            indent=2,
        )
    )

    print("=" * 72)
    print("WEATHER SHIFT SCENARIO")
    print("=" * 72)
    print(f"Delta:     {args.delta:+.1f}°C")
    print(f"Scenario:  {output}")
    print(f"Rows:      {len(weather)}")
    print(f"Schema:    {schema_path}")


if __name__ == "__main__":
    main()
