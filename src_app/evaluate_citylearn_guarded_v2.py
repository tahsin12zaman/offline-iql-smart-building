from pathlib import Path
import argparse
import csv
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CITYLEARN_ROOT = PROJECT_ROOT / "external" / "CityLearn"
if str(CITYLEARN_ROOT) not in sys.path:
    sys.path.insert(0, str(CITYLEARN_ROOT))

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.rbc import BasicRBC
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import IQLConfig

DEFAULT_SCHEMA = PROJECT_ROOT / "external" / "CityLearn" / "data" / "datasets" / "citylearn_challenge_2023_phase_2_local_evaluation" / "schema.json"
DEFAULT_DATA = PROJECT_ROOT / "data" / "citylearn_logged_multi.npz"
DEFAULT_MODEL = PROJECT_ROOT / "results" / "citylearn_iql_seed_1.d3"
DEFAULT_SUMMARY_OUT = PROJECT_ROOT / "results" / "citylearn_guarded_v2_summary.csv"
DEFAULT_EVENTS_OUT = PROJECT_ROOT / "results" / "citylearn_guarded_v2_events.csv"


def obs_to_array(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, list):
        obs = obs[0]
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def reward_to_float(reward):
    if isinstance(reward, list):
        return float(np.sum(reward))
    return float(np.asarray(reward).sum())


def series(building, attribute_name):
    return np.asarray(getattr(building, attribute_name), dtype=np.float64).reshape(-1)


def sum_building_metric(env, attribute_name):
    return sum(float(np.nansum(series(b, attribute_name))) for b in env.buildings)


def max_building_metric(env, attribute_name):
    values = []
    for b in env.buildings:
        values.extend(series(b, attribute_name).tolist())
    return float(np.nanmax(values)) if values else float("nan")


def build_iql():
    return IQLConfig(
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        expectile=0.7,
        weight_temp=3.0,
        max_weight=100.0,
    ).create(device=False)


def load_training_dataset(path):
    arr = np.load(path, allow_pickle=True)
    observations = arr["observations"].astype(np.float32)
    actions = arr["actions"].astype(np.float32)
    rewards = arr["rewards"].astype(np.float32)
    terminals = arr["terminals"].astype(bool)
    dataset = MDPDataset(observations=observations, actions=actions, rewards=rewards, terminals=terminals)
    return dataset, observations, actions


def load_iql_model(data_path, model_path):
    dataset, observations, actions = load_training_dataset(data_path)
    algo = build_iql()
    try:
        algo.build_with_dataset(dataset)
    except AttributeError:
        algo.create_impl(observation_shape=observations.shape[1:], action_size=actions.shape[1])
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Saved IQL model not found: {model_path}")
    algo.load_model(str(model_path))
    return algo


def calculate_building_comfort(building):
    indoor = series(building, "indoor_dry_bulb_temperature")
    setpoint = series(building, "indoor_dry_bulb_temperature_cooling_set_point")
    comfort_band = series(building, "comfort_band")
    occupants = series(building, "occupant_count")
    n = min(len(indoor), len(setpoint), len(comfort_band), len(occupants))
    if n == 0:
        return {"occupied_count": 0, "violation_count": 0, "discomfort": float("nan")}
    indoor, setpoint, comfort_band, occupants = indoor[:n], setpoint[:n], comfort_band[:n], occupants[:n]
    occupied = occupants > 0.0
    violations = occupied & (indoor > setpoint + comfort_band)
    occupied_count = int(np.sum(occupied))
    violation_count = int(np.sum(violations))
    discomfort = violation_count / occupied_count if occupied_count else float("nan")
    return {"occupied_count": occupied_count, "violation_count": violation_count, "discomfort": discomfort}


def collect_metrics(env, total_reward, steps):
    metrics = {
        "reward": float(total_reward),
        "steps": int(steps),
        "net_electricity_consumption": sum_building_metric(env, "net_electricity_consumption"),
        "electricity_cost": sum_building_metric(env, "net_electricity_consumption_cost"),
        "carbon_emission": sum_building_metric(env, "net_electricity_consumption_emission"),
        "peak_net_electricity": max_building_metric(env, "net_electricity_consumption"),
    }
    ds = []
    for i, b in enumerate(env.buildings, 1):
        c = calculate_building_comfort(b)
        metrics[f"building_{i}_occupied_count"] = c["occupied_count"]
        metrics[f"building_{i}_violation_count"] = c["violation_count"]
        metrics[f"building_{i}_overheating_discomfort"] = c["discomfort"]
        if np.isfinite(c["discomfort"]):
            ds.append(c["discomfort"])
    metrics["worst_building_overheating_discomfort"] = float(max(ds)) if ds else float("nan")
    return metrics


def latest_completed_value(building, attribute_name):
    arr = series(building, attribute_name)
    completed_index = int(building.time_step) - 1
    if completed_index < 0 or completed_index >= arr.size:
        return None
    value = float(arr[completed_index])
    return value if np.isfinite(value) else None


def get_completed_comfort_state(building):
    indoor = latest_completed_value(building, "indoor_dry_bulb_temperature")
    setpoint = latest_completed_value(building, "indoor_dry_bulb_temperature_cooling_set_point")
    comfort_band = latest_completed_value(building, "comfort_band")
    occupants = latest_completed_value(building, "occupant_count")
    if any(v is None for v in (indoor, setpoint, comfort_band, occupants)):
        return None
    upper = setpoint + comfort_band
    return {
        "completed_index": int(building.time_step) - 1,
        "indoor_temperature": indoor,
        "cooling_setpoint": setpoint,
        "comfort_band": comfort_band,
        "upper_comfort_boundary": upper,
        "occupant_count": occupants,
        "violation": bool(occupants > 0.0 and indoor > upper),
    }


def take_step(env, action):
    out = env.step([action])
    if len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        next_obs, reward, done, info = out
        done = bool(done)
    return next_obs, reward, done, info


def evaluate_normal(algo, schema, max_steps):
    env = CityLearnEnv(str(schema), central_agent=True)
    action_space = env.action_space[0]
    low, high = np.asarray(action_space.low, dtype=np.float32), np.asarray(action_space.high, dtype=np.float32)
    obs = obs_to_array(env.reset())
    total_reward, steps = 0.0, 0
    for _ in range(max_steps):
        action = np.clip(algo.predict(obs.reshape(1, -1))[0].astype(np.float32), low, high)
        next_obs, reward, done, _ = take_step(env, action)
        total_reward += reward_to_float(reward)
        steps += 1
        obs = obs_to_array(next_obs)
        if done:
            break
    metrics = collect_metrics(env, total_reward, steps)
    env.close()
    return metrics


def evaluate_guarded(algo, schema, max_steps):
    env = CityLearnEnv(str(schema), central_agent=True)
    rbc = BasicRBC(env)
    action_space = env.action_space[0]
    low, high = np.asarray(action_space.low, dtype=np.float32), np.asarray(action_space.high, dtype=np.float32)
    reset_out = env.reset()
    obs = obs_to_array(reset_out)
    observations_for_rbc = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    cooling_indices = [i for i, name in enumerate(env.action_names[0]) if name == "cooling_device"]
    if len(cooling_indices) != len(env.buildings):
        raise RuntimeError(f"Expected one cooling_device action per building; found {cooling_indices}")
    print("Cooling action indices:", cooling_indices)
    total_reward, steps, events = 0.0, 0, []
    counts = {i: 0 for i in range(1, len(env.buildings) + 1)}

    for loop_step in range(max_steps):
        learned = np.clip(algo.predict(obs.reshape(1, -1))[0].astype(np.float32), low, high)
        guarded = learned.copy()
        rbc_action = np.asarray(rbc.predict(observations_for_rbc, deterministic=True), dtype=np.float32).reshape(-1)
        if rbc_action.shape != guarded.shape:
            raise RuntimeError(f"BasicRBC and IQL shapes differ: RBC={rbc_action.shape}, IQL={guarded.shape}")
        rbc_action = np.clip(rbc_action, low, high)

        for building_index, building in enumerate(env.buildings, 1):
            state = get_completed_comfort_state(building)
            if state is None or not state["violation"]:
                continue
            idx = cooling_indices[building_index - 1]
            learned_cooling = float(guarded[idx])
            fallback_cooling = float(rbc_action[idx])
            guarded[idx] = fallback_cooling
            counts[building_index] += 1
            events.append({
                "loop_step": loop_step,
                "building": building_index,
                "building_time_step": int(building.time_step),
                "completed_index": state["completed_index"],
                "indoor_temperature": state["indoor_temperature"],
                "cooling_setpoint": state["cooling_setpoint"],
                "comfort_band": state["comfort_band"],
                "upper_comfort_boundary": state["upper_comfort_boundary"],
                "occupant_count": state["occupant_count"],
                "learned_cooling_action": learned_cooling,
                "fallback_cooling_action": fallback_cooling,
                "applied_cooling_action": fallback_cooling,
                "action_difference": fallback_cooling - learned_cooling,
            })

        guarded = np.clip(guarded, low, high)
        next_obs, reward, done, _ = take_step(env, guarded)
        total_reward += reward_to_float(reward)
        steps += 1
        observations_for_rbc = next_obs[0] if isinstance(next_obs, tuple) else next_obs
        obs = obs_to_array(next_obs)
        if done:
            break

    metrics = collect_metrics(env, total_reward, steps)
    metrics["guardrail_activations"] = len(events)
    for i in counts:
        metrics[f"building_{i}_guardrail_activations"] = counts[i]
    env.close()
    return metrics, events


def print_metrics(title, metrics):
    print("\n" + title)
    print("-" * 78)
    print(f"Control steps:             {metrics['steps']}")
    print(f"Cumulative reward:         {metrics['reward']:.6f}")
    print(f"Net electricity:           {metrics['net_electricity_consumption']:.6f}")
    print(f"Electricity cost:          {metrics['electricity_cost']:.6f}")
    print(f"Carbon emissions:          {metrics['carbon_emission']:.6f}")
    print(f"Peak building electricity: {metrics['peak_net_electricity']:.6f}")
    for i in range(1, 4):
        d = metrics[f"building_{i}_overheating_discomfort"]
        v = metrics[f"building_{i}_violation_count"]
        o = metrics[f"building_{i}_occupied_count"]
        text = f"{100*d:.2f}%" if np.isfinite(d) else "NaN"
        print(f"Building {i} overheating:   {text} ({v}/{o})")
    worst = metrics["worst_building_overheating_discomfort"]
    if np.isfinite(worst):
        print(f"Worst-building overheating: {100*worst:.2f}%")
    if "guardrail_activations" in metrics:
        print(f"Guardrail activations:      {metrics['guardrail_activations']}")
        for i in range(1, 4):
            key = f"building_{i}_guardrail_activations"
            print(f"  Building {i}:              {metrics.get(key, 0)}")


def write_summary_csv(path, normal, guarded):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set(normal) | set(guarded))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode"] + keys)
        w.writeheader()
        for mode, metrics in (("NORMAL_IQL", normal), ("GUARDED_IQL", guarded)):
            row = {"mode": mode}; row.update(metrics); w.writerow(row)


def write_events_csv(path, events):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        if not events:
            f.write("no_events\n"); return
        w = csv.DictWriter(f, fieldnames=list(events[0].keys()))
        w.writeheader(); w.writerows(events)


def pct_change(before, after):
    return (after - before) / abs(before) * 100.0 if np.isfinite(before) and before != 0 else float("nan")


def print_comparison(normal, guarded):
    print("\n" + "=" * 78)
    print("NORMAL vs GUARDED CHANGE")
    print("=" * 78)
    for label, key in [
        ("Net electricity", "net_electricity_consumption"),
        ("Electricity cost", "electricity_cost"),
        ("Carbon emissions", "carbon_emission"),
        ("Peak electricity", "peak_net_electricity"),
    ]:
        print(f"{label:24s}: {normal[key]:12.6f} -> {guarded[key]:12.6f} ({pct_change(normal[key], guarded[key]):+.2f}%)")
    print()
    for i in range(1, 4):
        key = f"building_{i}_overheating_discomfort"
        before, after = 100*normal[key], 100*guarded[key]
        print(f"Building {i} discomfort: {before:7.2f}% -> {after:7.2f}% ({after-before:+.2f} percentage points)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--max-steps", type=int, default=719)
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT))
    parser.add_argument("--events-out", default=str(DEFAULT_EVENTS_OUT))
    args = parser.parse_args()

    print("=" * 78)
    print("CityLearn Guarded IQL Evaluation V2")
    print("=" * 78)
    print(f"Schema: {args.schema}\nDataset: {args.data}\nModel:   {args.model}\nSteps:   {args.max_steps}")
    print("\nRuntime state: completed_index = building.time_step - 1")
    print("Condition: occupied AND indoor_temperature > cooling_setpoint + comfort_band")
    print("Action: replace only affected building's cooling_device action with BasicRBC cooling\n")

    print("Loading saved IQL controller...")
    algo = load_iql_model(args.data, args.model)
    print("Controller loaded.")

    print("\n" + "=" * 78 + "\nRUNNING NORMAL IQL\n" + "=" * 78)
    normal = evaluate_normal(algo, Path(args.schema), args.max_steps)
    print("\n" + "=" * 78 + "\nRUNNING GUARDED IQL\n" + "=" * 78)
    guarded, events = evaluate_guarded(algo, Path(args.schema), args.max_steps)

    print_metrics("NORMAL IQL", normal)
    print_metrics("GUARDED IQL", guarded)
    print_comparison(normal, guarded)

    print("\n" + "=" * 78 + "\nINTERVENTION SUMMARY\n" + "=" * 78)
    print(f"Total interventions: {len(events)}")
    for i in range(1, 4):
        print(f"Building {i}: {sum(e['building'] == i for e in events)}")
    if events:
        diffs = np.asarray([abs(e["action_difference"]) for e in events], dtype=np.float64)
        print(f"Mean absolute cooling-action difference: {np.mean(diffs):.6f}")
        print(f"Maximum absolute cooling-action difference: {np.max(diffs):.6f}")
        print(f"Non-zero action substitutions: {int(np.sum(diffs > 1e-6))}/{len(events)}")
        print("\nFirst 15 interventions:")
        for e in events[:15]:
            print(f"step={e['loop_step']:3d} B={e['building']} history={e['completed_index']:3d} T={e['indoor_temperature']:.3f} upper={e['upper_comfort_boundary']:.3f} occ={e['occupant_count']:.1f} IQL={e['learned_cooling_action']:.3f} RBC={e['fallback_cooling_action']:.3f}")

    write_summary_csv(args.summary_out, normal, guarded)
    write_events_csv(args.events_out, events)
    print("\n" + "=" * 78 + "\nSAVED OUTPUTS\n" + "=" * 78)
    print(f"Summary: {args.summary_out}\nEvents:  {args.events_out}")


if __name__ == "__main__":
    main()
