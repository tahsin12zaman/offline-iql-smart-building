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


DEFAULT_DATA = PROJECT_ROOT / "data" / "citylearn_logged_multi.npz"
DEFAULT_MODEL = PROJECT_ROOT / "results" / "citylearn_iql_seed_1.d3"

DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "results"
    / "citylearn_time_conditioned_reference.npz"
)

DEFAULT_ORIGINAL_SCHEMA = (
    PROJECT_ROOT
    / "external"
    / "CityLearn"
    / "data"
    / "datasets"
    / "citylearn_challenge_2023_phase_2_local_evaluation"
    / "schema.json"
)

DEFAULT_HOT_SCHEMA = (
    PROJECT_ROOT
    / "data"
    / "scenarios"
    / "hot_weather_plus3C"
    / "schema.json"
)


def obs_to_array(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, list):
        obs = obs[0]
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def build_iql():
    return IQLConfig(
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        expectile=0.7,
        weight_temp=3.0,
        max_weight=100.0,
    ).create(device=False)


def load_iql_model(data_path, model_path):
    arr = np.load(
        data_path,
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


def state_score(
    obs,
    timestep,
    time_mean,
    safe_time_std,
    active_mask,
):
    z = (
        obs - time_mean[timestep]
    ) / safe_time_std[timestep]

    z = z[active_mask]

    return float(
        np.sqrt(
            np.mean(
                np.square(z)
            )
        )
    )


def evaluate_scenario(
    algo,
    schema,
    scenario,
    max_steps,
    time_mean,
    safe_time_std,
    active_mask,
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

        horizon = len(time_mean)

        steps_to_run = min(
            max_steps,
            horizon,
        )

        for t in range(steps_to_run):

            score = state_score(
                obs,
                t,
                time_mean,
                safe_time_std,
                active_mask,
            )

            rows.append({
                "scenario": scenario,
                "step": t,
                "ood_score": score,
            })

            action = algo.predict(
                obs.reshape(1, -1)
            )[0].astype(np.float32)

            action = np.clip(
                action,
                low,
                high,
            )

            out = env.step([action])

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


def summarize(
    df,
    threshold_95,
    threshold_99,
):
    scores = df[
        "ood_score"
    ].to_numpy()

    return {
        "scenario": df[
            "scenario"
        ].iloc[0],
        "count": len(scores),
        "mean_score": float(
            np.mean(scores)
        ),
        "std_score": float(
            np.std(scores)
        ),
        "median_score": float(
            np.median(scores)
        ),
        "min_score": float(
            np.min(scores)
        ),
        "max_score": float(
            np.max(scores)
        ),
        "p90_score": float(
            np.quantile(scores, 0.90)
        ),
        "p95_score": float(
            np.quantile(scores, 0.95)
        ),
        "p99_score": float(
            np.quantile(scores, 0.99)
        ),
        "fraction_above_threshold_95":
            float(
                np.mean(
                    scores > threshold_95
                )
            ),
        "fraction_above_threshold_99":
            float(
                np.mean(
                    scores > threshold_99
                )
            ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
    )

    parser.add_argument(
        "--reference",
        default=str(DEFAULT_REFERENCE),
    )

    parser.add_argument(
        "--original-schema",
        default=str(DEFAULT_ORIGINAL_SCHEMA),
    )

    parser.add_argument(
        "--shifted-schema",
        default=str(DEFAULT_HOT_SCHEMA),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=719,
    )

    parser.add_argument(
        "--shift-label",
        default="+3C",
    )

    parser.add_argument(
        "--summary-out",
        default=(
            "results/"
            "citylearn_time_ood_summary.csv"
        ),
    )

    parser.add_argument(
        "--scores-out",
        default=(
            "results/"
            "citylearn_time_ood_scores.csv"
        ),
    )

    args = parser.parse_args()

    ref = np.load(
        args.reference,
        allow_pickle=True,
    )

    time_mean = ref[
        "time_mean"
    ].astype(np.float32)

    safe_time_std = ref[
        "safe_time_std"
    ].astype(np.float32)

    active_mask = ref[
        "active_feature_mask"
    ].astype(bool)

    threshold_95 = float(
        ref["threshold_95"]
    )

    threshold_99 = float(
        ref["threshold_99"]
    )

    print("=" * 80)
    print("TIME-CONDITIONED CITYLEARN OOD EVALUATION")
    print("=" * 80)

    print(
        f"Horizon:        {len(time_mean)}"
    )
    print(
        f"Active features:{int(active_mask.sum()):5d}"
    )
    print(
        f"P95 threshold:  {threshold_95:.9f}"
    )
    print(
        f"P99 threshold:  {threshold_99:.9f}"
    )

    print("\nLoading IQL...")
    algo = load_iql_model(
        args.data,
        args.model,
    )
    print("IQL loaded.")

    print(
        "\nEvaluating original weather..."
    )

    original = evaluate_scenario(
        algo,
        Path(args.original_schema),
        "Original",
        args.max_steps,
        time_mean,
        safe_time_std,
        active_mask,
    )

    print(
        "Evaluating +3°C weather..."
    )

    shifted = evaluate_scenario(
        algo,
        Path(args.shifted_schema),
        args.shift_label,
        args.max_steps,
        time_mean,
        safe_time_std,
        active_mask,
    )

    all_scores = pd.concat(
        [original, shifted],
        ignore_index=True,
    )

    all_scores[
        "above_threshold_95"
    ] = (
        all_scores["ood_score"]
        > threshold_95
    )

    all_scores[
        "above_threshold_99"
    ] = (
        all_scores["ood_score"]
        > threshold_99
    )

    summary = pd.DataFrame([
        summarize(
            original,
            threshold_95,
            threshold_99,
        ),
        summarize(
            shifted,
            threshold_95,
            threshold_99,
        ),
    ])

    labels = np.concatenate([
        np.zeros(
            len(original),
            dtype=int,
        ),
        np.ones(
            len(shifted),
            dtype=int,
        ),
    ])

    scores = np.concatenate([
        original[
            "ood_score"
        ].to_numpy(),
        shifted[
            "ood_score"
        ].to_numpy(),
    ])

    auroc = float(
        roc_auc_score(
            labels,
            scores,
        )
    )

    original_mean = float(
        original["ood_score"].mean()
    )

    shifted_mean = float(
        shifted["ood_score"].mean()
    )

    mean_change = (
        shifted_mean
        - original_mean
    )

    relative_change = (
        100.0
        * (
            shifted_mean
            / original_mean
            - 1.0
        )
        if original_mean != 0
        else float("nan")
    )

    Path(
        args.summary_out
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        args.summary_out,
        index=False,
    )

    all_scores.to_csv(
        args.scores_out,
        index=False,
    )

    print("\n" + "=" * 80)
    print("OOD SCORE SUMMARY")
    print("=" * 80)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print("\n" + "=" * 80)
    print("SHIFT SEPARATION")
    print("=" * 80)

    print(
        f"Original mean score: "
        f"{original_mean:.6f}"
    )

    print(
        f"{args.shift_label} mean score: "
        f"{shifted_mean:.6f}"
    )

    print(
        f"Mean change:         "
        f"{mean_change:.6f}"
    )

    print(
        f"Relative change:     "
        f"{relative_change:.2f}%"
    )

    print(
        f"AUROC:               "
        f"{auroc:.6f}"
    )

    print("\nSaved:")
    print(
        f"  {args.summary_out}"
    )
    print(
        f"  {args.scores_out}"
    )


if __name__ == "__main__":
    main()
