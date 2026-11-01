from pathlib import Path
import argparse
import csv
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CITYLEARN_ROOT = PROJECT_ROOT / "external" / "CityLearn"

if str(CITYLEARN_ROOT) not in sys.path:
    sys.path.insert(0, str(CITYLEARN_ROOT))

from citylearn.citylearn import CityLearnEnv


DEFAULT_REFERENCE = (
    PROJECT_ROOT / "results" / "citylearn_trust_reference.npz"
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


def positive_knn_scores(
    knn,
    queries,
    k,
    zero_tolerance=1e-6,
):
    distances, _ = knn.kneighbors(
        queries,
        return_distance=True,
    )

    scores = np.empty(len(queries), dtype=np.float64)

    for i, row in enumerate(distances):
        positive = row[row > zero_tolerance]

        if len(positive) < k:
            raise RuntimeError(
                f"Observation {i} has fewer than "
                f"{k} non-identical neighbours."
            )

        scores[i] = float(
            np.mean(positive[:k])
        )

    return scores


def collect_observations(schema, max_steps):
    env = CityLearnEnv(
        str(schema),
        central_agent=True,
    )

    try:
        reset_out = env.reset()
        obs = obs_to_array(reset_out)

        observations = [obs.copy()]

        # We use zero action because this experiment is testing
        # the observation distribution itself, not policy quality.
        action_space = env.action_space[0]

        zero_action = np.zeros(
            action_space.shape[0],
            dtype=np.float32,
        )

        zero_action = np.clip(
            zero_action,
            np.asarray(action_space.low, dtype=np.float32),
            np.asarray(action_space.high, dtype=np.float32),
        )

        steps = 0

        for _ in range(max_steps):
            out = env.step([zero_action])

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

                done = bool(np.any(done))

            obs = obs_to_array(next_obs)
            observations.append(obs.copy())

            steps += 1

            if done:
                break

        return np.asarray(
            observations,
            dtype=np.float32,
        )

    finally:
        env.close()


def score_observations(
    observations,
    mean,
    std,
    active_feature_mask,
    knn,
    k,
):
    if observations.shape[1] != len(mean):
        raise ValueError(
            f"Observation dimension mismatch: "
            f"{observations.shape[1]} vs {len(mean)}"
        )

    z = (
        observations - mean
    ) / std

    z = z[:, active_feature_mask]

    return positive_knn_scores(
        knn,
        z,
        k=k,
    )


def summarize_scores(
    scenario,
    scores,
    threshold_95,
    threshold_99,
):
    return {
        "scenario": scenario,
        "count": len(scores),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "median_score": float(np.median(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "p90_score": float(
            np.quantile(scores, 0.90)
        ),
        "p95_score": float(
            np.quantile(scores, 0.95)
        ),
        "p99_score": float(
            np.quantile(scores, 0.99)
        ),
        "fraction_above_threshold_95": float(
            np.mean(scores > threshold_95)
        ),
        "fraction_above_threshold_99": float(
            np.mean(scores > threshold_99)
        ),
    }


def main():
    parser = argparse.ArgumentParser()

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
        "--summary-out",
        default=(
            "results/"
            "citylearn_ood_original_vs_hot_summary.csv"
        ),
    )

    parser.add_argument(
        "--scores-out",
        default=(
            "results/"
            "citylearn_ood_original_vs_hot_scores.csv"
        ),
    )

    args = parser.parse_args()

    reference = np.load(
        args.reference,
        allow_pickle=True,
    )

    mean = reference["mean"].astype(
        np.float32
    )

    std = reference["std"].astype(
        np.float32
    )

    active_feature_mask = reference[
        "active_feature_mask"
    ].astype(bool)

    reference_standardized = reference[
        "reference_standardized"
    ].astype(np.float32)

    k = int(reference["k"])

    threshold_95 = float(
        reference["threshold_95"]
    )

    threshold_99 = float(
        reference["threshold_99"]
    )

    query_neighbors = min(
        len(reference_standardized),
        max(k + 64, 4 * k),
    )

    knn = NearestNeighbors(
        n_neighbors=query_neighbors,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )

    knn.fit(reference_standardized)

    print("=" * 80)
    print("CITYLEARN OOD EVALUATION")
    print("=" * 80)
    print(f"k:                {k}")
    print(
        f"95% threshold:    "
        f"{threshold_95:.6f}"
    )
    print(
        f"99% threshold:    "
        f"{threshold_99:.6f}"
    )

    print("\nCollecting original-weather observations...")

    original_obs = collect_observations(
        Path(args.original_schema),
        args.max_steps,
    )

    print(
        f"Original observations: "
        f"{len(original_obs)}"
    )

    print("\nCollecting +3°C observations...")

    shifted_obs = collect_observations(
        Path(args.shifted_schema),
        args.max_steps,
    )

    print(
        f"Shifted observations:  "
        f"{len(shifted_obs)}"
    )

    original_scores = score_observations(
        original_obs,
        mean,
        std,
        active_feature_mask,
        knn,
        k,
    )

    shifted_scores = score_observations(
        shifted_obs,
        mean,
        std,
        active_feature_mask,
        knn,
        k,
    )

    summaries = [
        summarize_scores(
            "Original",
            original_scores,
            threshold_95,
            threshold_99,
        ),
        summarize_scores(
            "+3C",
            shifted_scores,
            threshold_95,
            threshold_99,
        ),
    ]

    summary_df = pd.DataFrame(
        summaries
    )

    # AUROC:
    # 0 = original / in-distribution
    # 1 = shifted / OOD
    labels = np.concatenate([
        np.zeros(
            len(original_scores),
            dtype=int,
        ),
        np.ones(
            len(shifted_scores),
            dtype=int,
        ),
    ])

    all_scores = np.concatenate([
        original_scores,
        shifted_scores,
    ])

    auroc = float(
        roc_auc_score(
            labels,
            all_scores,
        )
    )

    score_df = pd.DataFrame({
        "scenario": (
            ["Original"]
            * len(original_scores)
            + ["+3C"]
            * len(shifted_scores)
        ),
        "step": list(
            range(len(original_scores))
        )
        + list(
            range(len(shifted_scores))
        ),
        "ood_score": all_scores,
    })

    score_df[
        "above_threshold_95"
    ] = (
        score_df["ood_score"]
        > threshold_95
    )

    score_df[
        "above_threshold_99"
    ] = (
        score_df["ood_score"]
        > threshold_99
    )

    summary_out = Path(
        args.summary_out
    )

    scores_out = Path(
        args.scores_out
    )

    summary_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        summary_out,
        index=False,
    )

    score_df.to_csv(
        scores_out,
        index=False,
    )

    print("\n" + "=" * 80)
    print("OOD SCORE SUMMARY")
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n" + "=" * 80)
    print("SHIFT SEPARATION")
    print("=" * 80)

    original_mean = float(original_scores.mean())
    shifted_mean = float(shifted_scores.mean())
    mean_increase = shifted_mean - original_mean

    print(
        f"Original mean score: {original_mean:.6f}"
    )

    print(
        f"+3°C mean score:     {shifted_mean:.6f}"
    )

    print(
        f"Mean increase:       {mean_increase:.6f}"
    )

    if original_mean != 0:
        relative_increase = 100.0 * (
            shifted_mean / original_mean - 1.0
        )

        print(
            f"Relative increase:   {relative_increase:.2f}%"
        )

    print(
        f"AUROC:               {auroc:.6f}"
    )

    print("\nSaved:")
    print(f"  {summary_out}")
    print(f"  {scores_out}")


if __name__ == "__main__":
    main()
