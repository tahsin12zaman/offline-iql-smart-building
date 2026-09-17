from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_APP = PROJECT_ROOT / "src_app"
if str(SRC_APP) not in sys.path:
    sys.path.insert(0, str(SRC_APP))

from citylearn.citylearn import CityLearnEnv
from evaluate_citylearn_models import load_dataset, build_and_load_algo, obs_to_array

DATA_PATH = PROJECT_ROOT / "data" / "citylearn_logged_multi.npz"
SCHEMA_PATH = PROJECT_ROOT / "external" / "CityLearn" / "data" / "datasets" / "citylearn_challenge_2023_phase_2_local_evaluation" / "schema.json"
MAX_STEPS = 719

def validate_files(algo_name, seed):
    model_path = PROJECT_ROOT / "results" / f"citylearn_{algo_name.lower()}_seed_{seed}.d3"
    missing = [str(p) for p in (DATA_PATH, SCHEMA_PATH, model_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Required evaluation files are missing:\n" + "\n".join(missing))
    return model_path

def reward_to_float(reward):
    return float(np.asarray(reward).sum())

def current_value(values, index):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(arr) == 0:
        return np.nan
    return float(arr[min(max(0, index), len(arr) - 1)])

def run_live_evaluation(algo_name, seed, max_steps=MAX_STEPS):
    algo_name = algo_name.lower()
    if algo_name not in {"bc", "iql", "cql"}:
        raise ValueError("Controller must be BC, IQL, or CQL.")
    if int(seed) not in {1, 2, 3}:
        raise ValueError("Seed must be 1, 2, or 3.")

    model_path = validate_files(algo_name, int(seed))
    dataset, observations, actions = load_dataset(str(DATA_PATH))
    algo = build_and_load_algo(
        algo_name=algo_name,
        model_path=str(model_path),
        dataset=dataset,
        observations=observations,
        actions=actions,
    )

    env = CityLearnEnv(str(SCHEMA_PATH), central_agent=True)
    try:
        space = env.action_space[0]
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        obs = obs_to_array(env.reset())
        total_reward = 0.0
        rows = []
        completed_steps = 0

        for step in range(max_steps):
            x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            action = np.clip(algo.predict(x)[0].astype(np.float32), low, high)
            result = env.step([action])

            if len(result) == 5:
                next_obs, reward, terminated, truncated, _ = result
                done = bool(np.any(terminated) or np.any(truncated))
            elif len(result) == 4:
                next_obs, reward, done, _ = result
                done = bool(np.any(done))
            else:
                raise RuntimeError("Unexpected CityLearn step output.")

            total_reward += reward_to_float(reward)
            completed_steps = step + 1
            idx = max(0, env.time_step - 1)
            district = 0.0

            for building_id, building in enumerate(env.buildings, 1):
                value = current_value(building.net_electricity_consumption, idx)
                district += value
                rows.append({
                    "algorithm": algo_name.upper(), "seed": int(seed),
                    "step": step + 1, "level": "building",
                    "building": building_id, "net_electricity": value,
                })

            rows.append({
                "algorithm": algo_name.upper(), "seed": int(seed),
                "step": step + 1, "level": "district",
                "building": 0, "net_electricity": district,
            })

            obs = obs_to_array(next_obs)
            if done:
                break

        net, cost, carbon, peaks = 0.0, 0.0, 0.0, []
        for building in env.buildings:
            n = np.asarray(building.net_electricity_consumption, dtype=np.float64)
            c = np.asarray(building.net_electricity_consumption_cost, dtype=np.float64)
            e = np.asarray(building.net_electricity_consumption_emission, dtype=np.float64)
            net += float(np.nansum(n))
            cost += float(np.nansum(c))
            carbon += float(np.nansum(e))
            peaks.extend(n.reshape(-1).tolist())

        metrics = {
            "algorithm": algo_name.upper(),
            "seed": int(seed),
            "reward": total_reward,
            "steps": completed_steps,
            "net_electricity_consumption": net,
            "electricity_cost": cost,
            "carbon_emission": carbon,
            "peak_net_electricity": float(np.nanmax(peaks)),
            "model_path": str(model_path),
        }
        return metrics, pd.DataFrame(rows)
    finally:
        env.close()
