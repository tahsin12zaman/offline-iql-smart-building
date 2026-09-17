#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

CONTROLLED_SEEDS = [1, 2, 3]
ALGORITHMS = ["bc", "iql", "cql"]

EVIDENCE_FILES = [
    "results/citylearn_application_metrics.csv",
    "results/citylearn_comfort_summary.csv",
    "results/citylearn_native_district_kpi_summary.csv",
    "results/citylearn_guarded_v2_summary.csv",
    "results/citylearn_guarded_v2_events.csv",
]

VERIFICATION_FILES = [
    "tests/test_application_acceptance.py",
    "tests/test_guardrail_evidence.py",
    "tests/test_live_evaluation.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(relative_path: str) -> dict:
    path = ROOT / relative_path

    if not path.is_file():
        raise FileNotFoundError(
            f"Required reproducibility artifact not found: {relative_path}"
        )

    return {
        "path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    saved_controllers = []

    for algorithm in ALGORITHMS:
        for seed in range(4):
            relative_path = (
                f"results/citylearn_{algorithm}_seed_{seed}.d3"
            )
            path = ROOT / relative_path

            if path.is_file():
                saved_controllers.append(
                    {
                        "algorithm": algorithm.upper(),
                        "seed": seed,
                        "path": relative_path,
                        "used_in_controlled_evaluation": (
                            seed in CONTROLLED_SEEDS
                        ),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )

    manifest = {
        "manifest_schema": "offline-iql-citylearn-experiment-manifest",
        "manifest_version": 1,
        "project": {
            "name": "Offline RL for Smart Building Energy Management",
            "application": "Streamlit engineering evaluation prototype",
        },
        "experiment_scope": {
            "environment": "CityLearn",
            "buildings": 3,
            "observation_features": 52,
            "continuous_actions": 9,
            "evaluation_horizon_steps": 719,
            "offline_logged_transitions": 21570,
            "learned_controllers": ["BC", "IQL", "CQL"],
            "conventional_reference": "BasicRBC",
            "controlled_evaluation_seeds": CONTROLLED_SEEDS,
        },
        "controller_artifacts": {
            "available_saved_controller_count": len(saved_controllers),
            "controlled_evaluation_artifact_count": sum(
                int(item["used_in_controlled_evaluation"])
                for item in saved_controllers
            ),
            "artifacts": saved_controllers,
        },
        "preserved_evidence": [
            file_record(path) for path in EVIDENCE_FILES
        ],
        "verification": {
            "test_files": [
                file_record(path) for path in VERIFICATION_FILES
            ],
            "acceptance_test_file":
                "tests/test_application_acceptance.py",
            "guardrail_regression_test_file":
                "tests/test_guardrail_evidence.py",
            "live_evaluation_test_file":
                "tests/test_live_evaluation.py",
        },
        "reproducibility": {
            "dependency_specification": "requirements.txt",
            "controller_execution_service": "app/citylearn_service.py",
            "manifest_generator":
                "scripts/generate_experiment_manifest.py",
            "result_directory": "results",
        },
        "validation_boundary": (
            "Evidence in this manifest supports reproducible "
            "simulation-based engineering evaluation in CityLearn. "
            "It does not establish physical-building safety, certify "
            "a controller for deployment, or replace BMS integration, "
            "commissioning, operator override, and field validation."
        ),
    }

    output = RESULTS / "experiment_manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {output.relative_to(ROOT)}")
    print(
        "Saved controller artifacts:",
        manifest["controller_artifacts"][
            "available_saved_controller_count"
        ],
    )
    print(
        "Controlled evaluation artifacts:",
        manifest["controller_artifacts"][
            "controlled_evaluation_artifact_count"
        ],
    )
    print(
        "Preserved evidence files:",
        len(manifest["preserved_evidence"]),
    )


if __name__ == "__main__":
    main()
