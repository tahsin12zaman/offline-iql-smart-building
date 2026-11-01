from pathlib import Path
import argparse
import json

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA = PROJECT_ROOT / "data" / "citylearn_logged_multi.npz"
DEFAULT_OUT = (
    PROJECT_ROOT
    / "results"
    / "citylearn_time_conditioned_reference.npz"
)
DEFAULT_META = (
    PROJECT_ROOT
    / "results"
    / "citylearn_time_conditioned_reference.json"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
    )

    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
    )

    parser.add_argument(
        "--meta-out",
        default=str(DEFAULT_META),
    )

    parser.add_argument(
        "--calibration-episodes",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    data = np.load(
        args.data,
        allow_pickle=True,
    )

    obs = data["observations"].astype(
        np.float32
    )

    episode_ids = data["episode_ids"].astype(
        np.int32
    )

    episodes = np.unique(episode_ids)

    lengths = [
        int(np.sum(episode_ids == e))
        for e in episodes
    ]

    if len(set(lengths)) != 1:
        raise ValueError(
            f"Episodes have different lengths: {lengths}"
        )

    horizon = lengths[0]

    print("=" * 78)
    print("TIME-CONDITIONED CITYLEARN OOD REFERENCE")
    print("=" * 78)
    print(f"Episodes:              {len(episodes)}")
    print(f"Episode horizon:       {horizon}")
    print(f"Observation dimension: {obs.shape[1]}")

    rng = np.random.default_rng(
        args.seed
    )

    shuffled = episodes.copy()
    rng.shuffle(shuffled)

    calibration_eps = np.sort(
        shuffled[:args.calibration_episodes]
    )

    reference_eps = np.sort(
        shuffled[args.calibration_episodes:]
    )

    def episode_matrix(ep):
        x = obs[episode_ids == ep]

        if len(x) != horizon:
            raise ValueError(
                f"Episode {ep} has {len(x)} observations."
            )

        return x

    reference = np.stack(
        [
            episode_matrix(ep)
            for ep in reference_eps
        ],
        axis=0,
    )

    calibration = np.stack(
        [
            episode_matrix(ep)
            for ep in calibration_eps
        ],
        axis=0,
    )

    # Shape:
    # reference = episodes x time x features
    time_mean = reference.mean(
        axis=0
    )

    time_std = reference.std(
        axis=0
    )

    # Global scale prevents deterministic features from
    # causing division-by-zero/infinite scores.
    global_std = reference.reshape(
        -1,
        reference.shape[-1],
    ).std(axis=0)

    scale_floor = np.maximum(
        0.05 * global_std,
        1e-3,
    )

    safe_time_std = np.maximum(
        time_std,
        scale_floor[None, :],
    )

    # Remove globally constant features.
    active_mask = global_std > 1e-8

    calibration_scores = []

    for ep in range(
        calibration.shape[0]
    ):
        z = (
            calibration[ep]
            - time_mean
        ) / safe_time_std

        z = z[:, active_mask]

        # RMS standardized deviation over features.
        scores = np.sqrt(
            np.mean(
                np.square(z),
                axis=1,
            )
        )

        calibration_scores.extend(
            scores.tolist()
        )

    calibration_scores = np.asarray(
        calibration_scores,
        dtype=np.float64,
    )

    threshold_95 = float(
        np.quantile(
            calibration_scores,
            0.95,
        )
    )

    threshold_99 = float(
        np.quantile(
            calibration_scores,
            0.99,
        )
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        out_path,
        time_mean=time_mean,
        time_std=time_std,
        safe_time_std=safe_time_std,
        global_std=global_std,
        active_feature_mask=active_mask,
        calibration_scores=calibration_scores,
        reference_episode_ids=reference_eps,
        calibration_episode_ids=calibration_eps,
        horizon=np.asarray(horizon),
        threshold_95=np.asarray(
            threshold_95
        ),
        threshold_99=np.asarray(
            threshold_99
        ),
    )

    metadata = {
        "total_episodes": int(
            len(episodes)
        ),
        "reference_episodes":
            reference_eps.tolist(),
        "calibration_episodes":
            calibration_eps.tolist(),
        "horizon": int(horizon),
        "observation_dimension": int(
            obs.shape[1]
        ),
        "active_features": int(
            active_mask.sum()
        ),
        "calibration_mean": float(
            calibration_scores.mean()
        ),
        "calibration_std": float(
            calibration_scores.std()
        ),
        "calibration_min": float(
            calibration_scores.min()
        ),
        "calibration_max": float(
            calibration_scores.max()
        ),
        "threshold_95": threshold_95,
        "threshold_99": threshold_99,
    }

    Path(args.meta_out).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print()
    print("Calibration scores")
    print("-" * 78)

    print(
        f"Mean: {calibration_scores.mean():.6f}"
    )
    print(
        f"Std:  {calibration_scores.std():.6f}"
    )
    print(
        f"Min:  {calibration_scores.min():.6f}"
    )
    print(
        f"Max:  {calibration_scores.max():.6f}"
    )
    print(
        f"P95:  {threshold_95:.6f}"
    )
    print(
        f"P99:  {threshold_99:.6f}"
    )

    print()
    print("Saved:")
    print(out_path)
    print(args.meta_out)


if __name__ == "__main__":
    main()
