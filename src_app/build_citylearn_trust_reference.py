from pathlib import Path
import argparse
import json

import numpy as np
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA = PROJECT_ROOT / "data" / "citylearn_logged_multi.npz"
DEFAULT_OUT = PROJECT_ROOT / "results" / "citylearn_trust_reference.npz"
DEFAULT_META = PROJECT_ROOT / "results" / "citylearn_trust_reference.json"


def standardize(x, mean, std):
    return (x - mean) / std


def positive_knn_scores(
    knn,
    queries,
    k,
    zero_tolerance=1e-6,
):
    """
    Compute mean distance to the first k non-identical neighbours.

    Repeated CityLearn episodes can contain exact copies of the
    same state. Zero-distance matches therefore contain no useful
    novelty information and are ignored.
    """
    distances, _ = knn.kneighbors(
        queries,
        return_distance=True,
    )

    scores = np.empty(len(queries), dtype=np.float64)

    insufficient = 0

    for i, row in enumerate(distances):
        positive = row[row > zero_tolerance]

        if len(positive) < k:
            scores[i] = np.nan
            insufficient += 1
        else:
            scores[i] = float(np.mean(positive[:k]))

    if insufficient:
        raise RuntimeError(
            f"{insufficient} observations had fewer than "
            f"{k} non-identical neighbours. Increase the "
            "number of queried neighbours."
        )

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Build an in-distribution trust/OOD reference from the CityLearn offline dataset."
    )

    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA),
    )

    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
    )

    parser.add_argument(
        "--meta-out",
        type=str,
        default=str(DEFAULT_META),
    )

    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbours used for OOD scoring.",
    )

    parser.add_argument(
        "--calibration-episodes",
        type=int,
        default=6,
        help="Number of whole episodes reserved for calibration.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    out_path = Path(args.out).resolve()
    meta_path = Path(args.meta_out).resolve()

    arr = np.load(data_path, allow_pickle=True)

    observations = arr["observations"].astype(np.float32)
    episode_ids = arr["episode_ids"].astype(np.int32)

    if observations.ndim != 2:
        raise ValueError(
            f"Expected 2D observations, got {observations.shape}"
        )

    if observations.shape[1] != 52:
        raise ValueError(
            f"Expected 52 observation dimensions, got {observations.shape[1]}"
        )

    if not np.isfinite(observations).all():
        raise ValueError("Dataset contains non-finite observations.")

    unique_episodes = np.unique(episode_ids)

    if args.calibration_episodes <= 0:
        raise ValueError("--calibration-episodes must be positive.")

    if args.calibration_episodes >= len(unique_episodes):
        raise ValueError(
            "Calibration episode count must be smaller than total episode count."
        )

    # ------------------------------------------------------------
    # Episode-level split
    # ------------------------------------------------------------

    rng = np.random.default_rng(args.seed)

    shuffled_episodes = unique_episodes.copy()
    rng.shuffle(shuffled_episodes)

    calibration_episode_ids = np.sort(
        shuffled_episodes[: args.calibration_episodes]
    )

    reference_episode_ids = np.sort(
        shuffled_episodes[args.calibration_episodes :]
    )

    reference_mask = np.isin(
        episode_ids,
        reference_episode_ids,
    )

    calibration_mask = np.isin(
        episode_ids,
        calibration_episode_ids,
    )

    reference_obs = observations[reference_mask]
    calibration_obs = observations[calibration_mask]

    # ------------------------------------------------------------
    # Normalization fitted ONLY on reference observations
    # ------------------------------------------------------------

    mean = reference_obs.mean(axis=0).astype(np.float32)
    raw_std = reference_obs.std(axis=0).astype(np.float32)

    constant_mask = raw_std < 1e-8

    safe_std = raw_std.copy()
    safe_std[constant_mask] = 1.0

    reference_z = standardize(
        reference_obs,
        mean,
        safe_std,
    ).astype(np.float32)

    calibration_z = standardize(
        calibration_obs,
        mean,
        safe_std,
    ).astype(np.float32)

    # Constant dimensions contain no OOD information.
    active_feature_mask = ~constant_mask

    reference_active = reference_z[:, active_feature_mask]
    calibration_active = calibration_z[:, active_feature_mask]

    # ------------------------------------------------------------
    # kNN OOD reference
    # ------------------------------------------------------------

    k = args.k

    if k <= 0:
        raise ValueError("--k must be positive.")

    if k > len(reference_active):
        raise ValueError(
            f"k={k} exceeds reference size {len(reference_active)}."
        )

    # We deliberately request substantially more than k neighbours
    # because corresponding timesteps from repeated episodes may be
    # exact duplicates.
    query_neighbors = min(
        len(reference_active),
        max(k + 64, 4 * k),
    )

    knn = NearestNeighbors(
        n_neighbors=query_neighbors,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )

    knn.fit(reference_active)

    calibration_scores = positive_knn_scores(
        knn,
        calibration_active,
        k=k,
        zero_tolerance=1e-6,
    )

    # Initial thresholds calibrated ONLY using held-out
    # in-distribution episodes.
    threshold_95 = float(
        np.quantile(calibration_scores, 0.95)
    )

    threshold_99 = float(
        np.quantile(calibration_scores, 0.99)
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        out_path,
        mean=mean,
        std=safe_std,
        raw_std=raw_std,
        constant_mask=constant_mask,
        active_feature_mask=active_feature_mask,
        reference_observations=reference_obs,
        reference_standardized=reference_active,
        calibration_observations=calibration_obs,
        calibration_scores=calibration_scores,
        reference_episode_ids=reference_episode_ids,
        calibration_episode_ids=calibration_episode_ids,
        k=np.asarray(k),
        threshold_95=np.asarray(threshold_95),
        threshold_99=np.asarray(threshold_99),
    )

    metadata = {
        "source_dataset": str(data_path),
        "total_observations": int(len(observations)),
        "observation_dimension": int(observations.shape[1]),
        "total_episodes": int(len(unique_episodes)),
        "reference_episodes": reference_episode_ids.tolist(),
        "calibration_episodes": calibration_episode_ids.tolist(),
        "reference_observations": int(len(reference_obs)),
        "calibration_observations": int(len(calibration_obs)),
        "constant_features": np.flatnonzero(
            constant_mask
        ).tolist(),
        "active_feature_count": int(
            np.sum(active_feature_mask)
        ),
        "k": int(k),
        "calibration_score_mean": float(
            calibration_scores.mean()
        ),
        "calibration_score_std": float(
            calibration_scores.std()
        ),
        "calibration_score_min": float(
            calibration_scores.min()
        ),
        "calibration_score_max": float(
            calibration_scores.max()
        ),
        "threshold_95": threshold_95,
        "threshold_99": threshold_99,
    }

    meta_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("CITYLEARN TRUST / OOD REFERENCE")
    print("=" * 78)

    print(f"Source dataset:          {data_path}")
    print(f"Total observations:      {len(observations)}")
    print(f"Observation dimension:   {observations.shape[1]}")
    print(f"Total episodes:          {len(unique_episodes)}")

    print(
        f"Reference episodes:      "
        f"{len(reference_episode_ids)}"
    )

    print(
        f"Calibration episodes:    "
        f"{len(calibration_episode_ids)}"
    )

    print(
        f"Reference observations:  "
        f"{len(reference_obs)}"
    )

    print(
        f"Calibration observations:"
        f"  {len(calibration_obs)}"
    )

    print(
        f"Constant features:       "
        f"{int(constant_mask.sum())}"
    )

    print(
        f"Active OOD features:     "
        f"{int(active_feature_mask.sum())}"
    )

    print(f"k nearest neighbours:    {k}")

    print("\nCalibration OOD scores")
    print("-" * 78)

    print(
        f"Mean:                     "
        f"{calibration_scores.mean():.9e}"
    )

    print(
        f"Std:                      "
        f"{calibration_scores.std():.9e}"
    )

    print(
        f"Min:                      "
        f"{calibration_scores.min():.9e}"
    )

    print(
        f"Max:                      "
        f"{calibration_scores.max():.9e}"
    )

    print(
        f"95th percentile:          "
        f"{threshold_95:.9e}"
    )

    print(
        f"99th percentile:          "
        f"{threshold_99:.9e}"
    )

    print("\nSaved:")
    print(f"  {out_path}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()
