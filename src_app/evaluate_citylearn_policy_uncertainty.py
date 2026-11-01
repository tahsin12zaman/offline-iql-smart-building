from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CITYLEARN_ROOT = PROJECT_ROOT / "external" / "CityLearn"

if str(CITYLEARN_ROOT) not in sys.path:
    sys.path.insert(0, str(CITYLEARN_ROOT))

from citylearn.citylearn import CityLearnEnv
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import IQLConfig


DEFAULT_DATA = (
    PROJECT_ROOT
    / "data"
    / "citylearn_logged_multi.npz"
)

DEFAULT_ANCHOR_MODEL = (
    PROJECT_ROOT
    / "results"
    / "citylearn_iql_seed_1.d3"
)

DEFAULT_ENSEMBLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "trust_ensemble"
)

ORIGINAL_SCHEMA = (
    PROJECT_ROOT
    / "external"
    / "CityLearn"
    / "data"
    / "datasets"
    / "citylearn_challenge_2023_phase_2_local_evaluation"
    / "schema.json"
)

SCENARIOS = [
    (
        "Original",
        0.0,
        ORIGINAL_SCHEMA,
    ),
    (
        "+1C",
        1.0,
        PROJECT_ROOT
        / "data"
        / "scenarios"
        / "hot_weather_plus1C"
        / "schema.json",
    ),
    (
        "+2C",
        2.0,
        PROJECT_ROOT
        / "data"
        / "scenarios"
        / "hot_weather_plus2C"
        / "schema.json",
    ),
    (
        "+3C",
        3.0,
        PROJECT_ROOT
        / "data"
        / "scenarios"
        / "hot_weather_plus3C"
        / "schema.json",
    ),
]


def obs_to_array(obs):
    if isinstance(obs, tuple):
        obs = obs[0]

    if isinstance(obs, list):
        obs = obs[0]

    return np.asarray(
        obs,
        dtype=np.float32,
    ).reshape(-1)


def build_iql():
    return IQLConfig(
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        expectile=0.7,
        weight_temp=3.0,
        max_weight=100.0,
    ).create(device=False)


def load_dataset(path):
    arr = np.load(
        path,
        allow_pickle=True,
    )

    observations = arr[
        "observations"
    ].astype(np.float32)

    actions = arr[
        "actions"
    ].astype(np.float32)

    rewards = arr[
        "rewards"
    ].astype(np.float32)

    terminals = arr[
        "terminals"
    ].astype(bool)

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    return (
        dataset,
        observations,
        actions,
    )


def load_iql(
    dataset,
    observations,
    actions,
    model_path,
):
    algo = build_iql()

    try:
        algo.build_with_dataset(dataset)
    except AttributeError:
        algo.create_impl(
            observation_shape=observations.shape[1:],
            action_size=actions.shape[1],
        )

    algo.load_model(
        str(model_path)
    )

    return algo


def ensemble_statistics(
    models,
    obs,
    low,
    high,
):
    predicted = []

    for model in models:
        action = model.predict(
            obs.reshape(1, -1)
        )[0].astype(np.float32)

        action = np.clip(
            action,
            low,
            high,
        )

        predicted.append(action)

    action_matrix = np.stack(
        predicted,
        axis=0,
    )

    raw_variance = np.var(
        action_matrix,
        axis=0,
    )

    raw_uncertainty = float(
        np.mean(raw_variance)
    )

    action_range = high - low

    safe_range = np.where(
        np.abs(action_range) > 1e-8,
        action_range,
        1.0,
    )

    normalized_actions = (
        action_matrix - low[None, :]
    ) / safe_range[None, :]

    normalized_variance = np.var(
        normalized_actions,
        axis=0,
    )

    normalized_std = np.std(
        normalized_actions,
        axis=0,
    )

    uncertainty = float(
        np.mean(
            normalized_variance
        )
    )

    rms_disagreement = float(
        np.sqrt(uncertainty)
    )

    max_dimension_std = float(
        np.max(normalized_std)
    )

    return {
        "uncertainty": uncertainty,
        "rms_disagreement": rms_disagreement,
        "raw_uncertainty": raw_uncertainty,
        "max_dimension_std": max_dimension_std,
        "dimension_std": normalized_std,
    }


def evaluate_scenario(
    anchor,
    ensemble,
    scenario,
    shift_c,
    schema,
    max_steps,
):
    env = CityLearnEnv(
        str(schema),
        central_agent=True,
    )

    try:
        action_space = env.action_space[0]

        low = np.asarray(
            action_space.low,
            dtype=np.float32,
        )

        high = np.asarray(
            action_space.high,
            dtype=np.float32,
        )

        obs = obs_to_array(
            env.reset()
        )

        rows = []

        for step in range(max_steps):
            stats = ensemble_statistics(
                ensemble,
                obs,
                low,
                high,
            )

            row = {
                "scenario": scenario,
                "shift_c": shift_c,
                "step": step,
                "policy_uncertainty":
                    stats["uncertainty"],
                "rms_disagreement":
                    stats["rms_disagreement"],
                "raw_uncertainty":
                    stats["raw_uncertainty"],
                "max_dimension_std":
                    stats["max_dimension_std"],
            }

            for i, value in enumerate(
                stats["dimension_std"]
            ):
                row[
                    f"action_{i}_ensemble_std"
                ] = float(value)

            rows.append(row)

            anchor_action = anchor.predict(
                obs.reshape(1, -1)
            )[0].astype(np.float32)

            anchor_action = np.clip(
                anchor_action,
                low,
                high,
            )

            out = env.step(
                [anchor_action]
            )

            if len(out) == 5:
                (
                    next_obs,
                    _,
                    terminated,
                    truncated,
                    _,
                ) = out

                done = bool(
                    np.any(terminated)
                    or np.any(truncated)
                )
            else:
                (
                    next_obs,
                    _,
                    done,
                    _,
                ) = out

                done = bool(
                    np.any(done)
                )

            obs = obs_to_array(
                next_obs
            )

            if done:
                break

        return pd.DataFrame(rows)

    finally:
        env.close()


def summarize(df):
    scores = df[
        "policy_uncertainty"
    ].to_numpy()

    rms = df[
        "rms_disagreement"
    ].to_numpy()

    maximum = df[
        "max_dimension_std"
    ].to_numpy()

    return {
        "scenario":
            df["scenario"].iloc[0],
        "shift_c":
            float(df["shift_c"].iloc[0]),
        "count":
            len(df),
        "mean_uncertainty":
            float(np.mean(scores)),
        "std_uncertainty":
            float(np.std(scores)),
        "median_uncertainty":
            float(np.median(scores)),
        "p90_uncertainty":
            float(np.quantile(scores, 0.90)),
        "p95_uncertainty":
            float(np.quantile(scores, 0.95)),
        "p99_uncertainty":
            float(np.quantile(scores, 0.99)),
        "max_uncertainty":
            float(np.max(scores)),
        "mean_rms_disagreement":
            float(np.mean(rms)),
        "mean_max_dimension_std":
            float(np.mean(maximum)),
    }


def calculate_auroc(
    original,
    shifted,
):
    original_scores = original[
        "policy_uncertainty"
    ].to_numpy()

    shifted_scores = shifted[
        "policy_uncertainty"
    ].to_numpy()

    labels = np.concatenate([
        np.zeros(
            len(original_scores),
            dtype=np.int32,
        ),
        np.ones(
            len(shifted_scores),
            dtype=np.int32,
        ),
    ])

    scores = np.concatenate([
        original_scores,
        shifted_scores,
    ])

    return float(
        roc_auc_score(
            labels,
            scores,
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
    )

    parser.add_argument(
        "--anchor-model",
        default=str(DEFAULT_ANCHOR_MODEL),
    )

    parser.add_argument(
        "--ensemble-dir",
        default=str(DEFAULT_ENSEMBLE_DIR),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=719,
    )

    parser.add_argument(
        "--scores-out",
        default=(
            "results/"
            "citylearn_policy_uncertainty_scores.csv"
        ),
    )

    parser.add_argument(
        "--summary-out",
        default=(
            "results/"
            "citylearn_policy_uncertainty_summary.csv"
        ),
    )

    parser.add_argument(
        "--auroc-out",
        default=(
            "results/"
            "citylearn_policy_uncertainty_auroc.csv"
        ),
    )

    args = parser.parse_args()

    dataset, observations, actions = load_dataset(
        args.data
    )

    ensemble_dir = Path(
        args.ensemble_dir
    )

    ensemble_paths = sorted(
        ensemble_dir.glob(
            "citylearn_iql_seed_*.d3"
        )
    )

    if len(ensemble_paths) != 5:
        raise RuntimeError(
            "Expected exactly 5 ensemble models, "
            f"found {len(ensemble_paths)}: "
            f"{ensemble_paths}"
        )

    print("=" * 80)
    print("CITYLEARN IQL POLICY-UNCERTAINTY SWEEP")
    print("=" * 80)

    print(f"Dataset:          {args.data}")
    print(f"Anchor model:     {args.anchor_model}")
    print(f"Ensemble members: {len(ensemble_paths)}")

    for path in ensemble_paths:
        print(f"  {path}")

    print("\nLoading anchor model...")

    anchor = load_iql(
        dataset,
        observations,
        actions,
        Path(args.anchor_model),
    )

    print("Anchor loaded.")

    ensemble = []

    for path in ensemble_paths:
        print(
            f"Loading ensemble member: {path.name}"
        )

        ensemble.append(
            load_iql(
                dataset,
                observations,
                actions,
                path,
            )
        )

    print("\nAll ensemble members loaded.")

    scenario_frames = {}

    for (
        label,
        shift_c,
        schema,
    ) in SCENARIOS:

        if not Path(schema).exists():
            raise FileNotFoundError(
                f"Missing schema for {label}: "
                f"{schema}"
            )

        print()
        print("-" * 80)
        print(
            f"Evaluating {label} "
            f"(shift={shift_c:+.1f}°C)"
        )
        print("-" * 80)

        frame = evaluate_scenario(
            anchor=anchor,
            ensemble=ensemble,
            scenario=label,
            shift_c=shift_c,
            schema=schema,
            max_steps=args.max_steps,
        )

        scenario_frames[label] = frame

        print(f"Steps: {len(frame)}")

        print(
            "Mean uncertainty: "
            f"{frame['policy_uncertainty'].mean():.9f}"
        )

        print(
            "Mean RMS disagreement: "
            f"{frame['rms_disagreement'].mean():.9f}"
        )

    all_scores = pd.concat(
        list(
            scenario_frames.values()
        ),
        ignore_index=True,
    )

    summary = pd.DataFrame([
        summarize(frame)
        for frame in scenario_frames.values()
    ])

    original = scenario_frames[
        "Original"
    ]

    auroc_rows = []

    for label in [
        "+1C",
        "+2C",
        "+3C",
    ]:
        shifted = scenario_frames[
            label
        ]

        auroc = calculate_auroc(
            original,
            shifted,
        )

        original_mean = float(
            original[
                "policy_uncertainty"
            ].mean()
        )

        shifted_mean = float(
            shifted[
                "policy_uncertainty"
            ].mean()
        )

        ratio = (
            shifted_mean
            / original_mean
            if original_mean > 0.0
            else float("nan")
        )

        auroc_rows.append({
            "scenario": label,
            "mean_uncertainty_original":
                original_mean,
            "mean_uncertainty_shifted":
                shifted_mean,
            "shift_to_original_ratio":
                ratio,
            "auroc":
                auroc,
        })

    auroc_df = pd.DataFrame(
        auroc_rows
    )

    means = summary[
        "mean_uncertainty"
    ].to_numpy()

    shifts = summary[
        "shift_c"
    ].to_numpy()

    if (
        np.std(means) > 0.0
        and np.std(shifts) > 0.0
    ):
        severity_correlation = float(
            np.corrcoef(
                shifts,
                means,
            )[0, 1]
        )
    else:
        severity_correlation = float(
            "nan"
        )

    scores_out = Path(
        args.scores_out
    )

    summary_out = Path(
        args.summary_out
    )

    auroc_out = Path(
        args.auroc_out
    )

    scores_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_scores.to_csv(
        scores_out,
        index=False,
    )

    summary.to_csv(
        summary_out,
        index=False,
    )

    auroc_df.to_csv(
        auroc_out,
        index=False,
    )

    print()
    print("=" * 80)
    print("POLICY UNCERTAINTY SUMMARY")
    print("=" * 80)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.9f}"
            ),
        )
    )

    print()
    print("=" * 80)
    print("SHIFT SEPARATION")
    print("=" * 80)

    print(
        auroc_df.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print()
    print(
        "Correlation between temperature "
        "shift magnitude and mean uncertainty: "
        f"{severity_correlation:.6f}"
    )

    print()
    print("Saved:")
    print(f"  {scores_out}")
    print(f"  {summary_out}")
    print(f"  {auroc_out}")


if __name__ == "__main__":
    main()