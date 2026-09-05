from pathlib import Path
from citylearn.citylearn import CityLearnEnv

DATASETS_DIR = Path("external/CityLearn/data/datasets")


def main():
    print("=" * 80)
    print("CityLearn Dataset Action Space Check")
    print("=" * 80)

    for schema in sorted(DATASETS_DIR.glob("*/schema.json")):
        dataset_name = schema.parent.name

        try:
            env = CityLearnEnv(str(schema.resolve()), central_agent=True)
            print(f"{dataset_name} => {env.action_space}")
        except Exception as e:
            print(f"{dataset_name} => ERROR: {str(e).splitlines()[0]}")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()