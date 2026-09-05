from pathlib import Path
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.base import BaselineAgent


def main():
    schema_path = Path(
        "external/CityLearn/data/datasets/citylearn_challenge_2023_phase_2_local_evaluation/schema.json"
    ).resolve()

    print("=" * 70)
    print("CityLearn HVAC/Energy Control Smoke Test")
    print("=" * 70)
    print("Schema:", schema_path)

    env = CityLearnEnv(str(schema_path), central_agent=True)
    agent = BaselineAgent(env)

    reset_out = env.reset()
    observations = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    total_reward = 0.0
    steps = 0

    while steps < 24:
        actions = agent.predict(observations)
        step_out = env.step(actions)

        if len(step_out) == 5:
            observations, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            observations, reward, done, info = step_out

        if isinstance(reward, list):
            total_reward += sum(reward)
        else:
            total_reward += float(reward)

        steps += 1

        if done:
            break

    print("Ran steps:", steps)
    print("Total reward over 24 steps:", total_reward)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    print("=" * 70)
    print("Smoke test completed.")


if __name__ == "__main__":
    main()