from pathlib import Path
from citylearn.citylearn import CityLearnEnv


SCHEMA = (
    "external/CityLearn/data/datasets/"
    "citylearn_challenge_2023_phase_2_local_evaluation/schema.json"
)


def main():
    schema = Path(SCHEMA).resolve()

    env = CityLearnEnv(
        str(schema),
        central_agent=True,
    )

    print("=" * 70)
    print("CityLearn Application Metric Inspection")
    print("=" * 70)

    print("Buildings:", len(env.buildings))
    print("Time steps:", env.time_steps)

    print("\nevaluate() available:", hasattr(env, "evaluate"))

    print("\n=== Building 0 potentially useful attributes ===")

    building = env.buildings[0]

    keywords = [
        "energy",
        "electric",
        "cost",
        "carbon",
        "comfort",
        "temperature",
        "demand",
        "consumption",
        "power",
    ]

    for name in dir(building):
        if name.startswith("_"):
            continue

        lower = name.lower()

        if any(keyword in lower for keyword in keywords):
            print(name)

    print("=" * 70)


if __name__ == "__main__":
    main()