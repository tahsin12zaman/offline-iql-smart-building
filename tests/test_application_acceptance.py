"""Application-level acceptance checks for the smart-building engineering prototype.

These tests verify that the preserved evidence package, saved controller artifacts,
and user-facing engineering application remain internally consistent.

They validate the documented CityLearn simulation/application scope only. They do
not constitute physical-building certification or deployment validation.
"""

from pathlib import Path
import ast

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "app.py"
DATA = ROOT / "data"
RESULTS = ROOT / "results"

APPLICATION_METRICS = RESULTS / "citylearn_application_metrics.csv"
COMFORT_SUMMARY = RESULTS / "citylearn_comfort_summary.csv"
NATIVE_KPI_SUMMARY = RESULTS / "citylearn_native_district_kpi_summary.csv"
GUARD_SUMMARY = RESULTS / "citylearn_guarded_v2_summary.csv"
GUARD_EVENTS = RESULTS / "citylearn_guarded_v2_events.csv"

EXPECTED_ALGORITHMS = {"BC", "IQL", "CQL"}
EXPECTED_SEEDS = {1, 2, 3}
EXPECTED_HORIZON = 719


def _read(path):
    assert path.exists(), f"Missing required evidence artifact: {path}"
    df = pd.read_csv(path)
    assert not df.empty, f"Evidence artifact is empty: {path}"
    return df


def test_primary_offline_dataset_exists():
    path = DATA / "citylearn_logged_multi.npz"
    assert path.exists(), f"Missing primary offline dataset: {path}"
    assert path.stat().st_size > 0


def test_expected_saved_controller_artifacts_exist():
    missing = []

    for algorithm in ("bc", "iql", "cql"):
        for seed in EXPECTED_SEEDS:
            path = RESULTS / f"citylearn_{algorithm}_seed_{seed}.d3"
            if not path.exists() or path.stat().st_size == 0:
                missing.append(str(path.relative_to(ROOT)))

    assert not missing, "Missing saved controller artifacts: " + ", ".join(missing)


def test_required_engineering_evidence_files_are_readable():
    for path in (
        APPLICATION_METRICS,
        COMFORT_SUMMARY,
        NATIVE_KPI_SUMMARY,
        GUARD_SUMMARY,
        GUARD_EVENTS,
    ):
        _read(path)


def test_application_metrics_schema_and_controlled_runs():
    df = _read(APPLICATION_METRICS)

    required_columns = {
        "algorithm",
        "seed",
        "reward",
        "steps",
        "net_electricity_consumption",
        "electricity_cost",
        "carbon_emission",
        "peak_net_electricity",
    }
    assert required_columns.issubset(df.columns)

    controlled = df[
        df["algorithm"].isin(EXPECTED_ALGORITHMS)
        & df["seed"].isin(EXPECTED_SEEDS)
    ].copy()

    expected_pairs = {
        (algorithm, seed)
        for algorithm in EXPECTED_ALGORITHMS
        for seed in EXPECTED_SEEDS
    }
    actual_pairs = set(
        zip(
            controlled["algorithm"].astype(str),
            controlled["seed"].astype(int),
        )
    )

    assert actual_pairs == expected_pairs
    assert len(controlled) == 9
    assert (controlled["steps"].astype(int) == EXPECTED_HORIZON).all()


def test_operational_metrics_are_finite_and_nonnegative():
    df = _read(APPLICATION_METRICS)

    for column in (
        "net_electricity_consumption",
        "electricity_cost",
        "carbon_emission",
        "peak_net_electricity",
    ):
        values = pd.to_numeric(df[column], errors="coerce")
        assert values.notna().all(), f"Non-numeric values found in {column}"
        assert (values >= 0).all(), f"Negative values found in {column}"


def test_comfort_evidence_covers_all_learned_controllers_and_buildings():
    df = _read(COMFORT_SUMMARY)

    required_columns = {"algorithm", "name", "cost_function", "mean", "std"}
    assert required_columns.issubset(df.columns)

    assert EXPECTED_ALGORITHMS.issubset(set(df["algorithm"]))

    discomfort = df[df["cost_function"] == "discomfort_proportion"]
    assert not discomfort.empty

    expected_buildings = {"Building_1", "Building_2", "Building_3"}

    for algorithm in EXPECTED_ALGORITHMS:
        subset = discomfort[discomfort["algorithm"] == algorithm]
        assert expected_buildings.issubset(set(subset["name"]))


def test_native_kpi_evidence_covers_all_learned_controllers():
    df = _read(NATIVE_KPI_SUMMARY)

    required_columns = {"algorithm", "cost_function", "mean", "std"}
    assert required_columns.issubset(df.columns)
    assert EXPECTED_ALGORITHMS.issubset(set(df["algorithm"]))


def test_guardrail_summary_and_events_are_consistent():
    summary = _read(GUARD_SUMMARY)
    events = _read(GUARD_EVENTS)

    assert "mode" in summary.columns
    assert {"NORMAL_IQL", "GUARDED_IQL"}.issubset(set(summary["mode"]))

    normal = summary[summary["mode"] == "NORMAL_IQL"].iloc[0]
    guarded = summary[summary["mode"] == "GUARDED_IQL"].iloc[0]

    assert int(normal["steps"]) == EXPECTED_HORIZON
    assert int(guarded["steps"]) == EXPECTED_HORIZON

    assert int(guarded["guardrail_activations"]) == len(events)
    assert len(events) == 207

    assert float(guarded["worst_building_overheating_discomfort"]) < float(
        normal["worst_building_overheating_discomfort"]
    )


def test_iql_seed_1_guardrail_baseline_matches_application_metrics():
    metrics = _read(APPLICATION_METRICS)
    summary = _read(GUARD_SUMMARY)

    iql = metrics[
        (metrics["algorithm"] == "IQL")
        & (metrics["seed"].astype(int) == 1)
    ].iloc[0]

    normal = summary[summary["mode"] == "NORMAL_IQL"].iloc[0]

    for column in (
        "reward",
        "net_electricity_consumption",
        "electricity_cost",
        "carbon_emission",
        "peak_net_electricity",
    ):
        assert float(normal[column]) == pytest.approx(
            float(iql[column]),
            rel=1e-8,
            abs=1e-8,
        )


def test_application_python_is_syntactically_valid():
    source = APP.read_text(encoding="utf-8")
    ast.parse(source)


def test_application_contains_guided_engineering_scenario():
    source = APP.read_text(encoding="utf-8")

    required_markers = (
        '"Engineering Scenario"',
        'st.subheader("Guided Engineering Scenario")',
        'st.markdown("### Engineering workflow")',
        'st.markdown("### Demonstration case — IQL Seed 1")',
        'st.markdown("### What this scenario demonstrates")',
    )

    for marker in required_markers:
        assert marker in source, f"Missing application scenario marker: {marker}"


def test_application_contains_full_requirements_traceability():
    source = APP.read_text(encoding="utf-8")

    for requirement_id in range(1, 10):
        marker = f'"ID": "R{requirement_id}"'
        assert marker in source, f"Missing traceability requirement R{requirement_id}"

    for marker in (
        '"Verification method"',
        '"Evidence / artifact"',
        '"Status": "VERIFIED"',
        "Traceability status refers to verification",
    ):
        assert marker in source


def test_application_contains_acceptance_and_evidence_generation():
    source = APP.read_text(encoding="utf-8")

    required_markers = (
        'st.markdown("### Engineering acceptance screening")',
        'st.markdown("### Controller evaluation report")',
        '"Download Controller Evaluation Report"',
        'st.markdown("### Controller verification record")',
        '"Download Controller Verification Record"',
        'st.markdown("### Reproducibility specification")',
        'st.markdown("### Failure and safety analysis")',
        'st.markdown("### Deployment boundary")',
    )

    for marker in required_markers:
        assert marker in source, f"Missing engineering application capability: {marker}"


def test_experiment_manifest_is_valid_and_consistent():
    import hashlib
    import json

    manifest_path = RESULTS / "experiment_manifest.json"
    generator_path = ROOT / "scripts" / "generate_experiment_manifest.py"

    assert manifest_path.exists(), "Missing experiment manifest"
    assert generator_path.exists(), "Missing experiment manifest generator"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert (
        manifest["manifest_schema"]
        == "offline-iql-citylearn-experiment-manifest"
    )
    assert manifest["manifest_version"] == 1

    scope = manifest["experiment_scope"]
    assert scope["environment"] == "CityLearn"
    assert scope["buildings"] == 3
    assert scope["observation_features"] == 52
    assert scope["continuous_actions"] == 9
    assert scope["evaluation_horizon_steps"] == EXPECTED_HORIZON
    assert scope["offline_logged_transitions"] == 21570
    assert set(scope["controlled_evaluation_seeds"]) == EXPECTED_SEEDS
    assert set(scope["learned_controllers"]) == EXPECTED_ALGORITHMS
    assert scope["conventional_reference"] == "BasicRBC"

    controller_artifacts = manifest["controller_artifacts"]
    assert controller_artifacts["available_saved_controller_count"] == 12
    assert controller_artifacts["controlled_evaluation_artifact_count"] == 9

    controlled = [
        artifact
        for artifact in controller_artifacts["artifacts"]
        if artifact["used_in_controlled_evaluation"]
    ]
    assert len(controlled) == 9

    for artifact in controller_artifacts["artifacts"]:
        path = ROOT / artifact["path"]
        assert path.exists()
        assert path.stat().st_size == artifact["size_bytes"]

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == artifact["sha256"]

    expected_evidence = {
        "results/citylearn_application_metrics.csv",
        "results/citylearn_comfort_summary.csv",
        "results/citylearn_native_district_kpi_summary.csv",
        "results/citylearn_guarded_v2_summary.csv",
        "results/citylearn_guarded_v2_events.csv",
    }

    evidence_paths = {
        item["path"] for item in manifest["preserved_evidence"]
    }
    assert evidence_paths == expected_evidence

    for artifact in manifest["preserved_evidence"]:
        path = ROOT / artifact["path"]
        assert path.exists()
        assert path.stat().st_size == artifact["size_bytes"]

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == artifact["sha256"]


def test_application_exposes_experiment_manifest():
    source = APP.read_text(encoding="utf-8")

    required_markers = (
        'experiment_manifest = read_json("experiment_manifest.json")',
        'st.markdown("#### Experiment manifest")',
        '"Download Experiment Manifest (JSON)"',
        '"Inspect manifest provenance"',
        '"download_experiment_manifest"',
        'read_json("experiment_manifest.json")',
        '["manifest_generator"]',
    )

    for marker in required_markers:
        assert marker in source, (
            f"Missing experiment-manifest application marker: {marker}"
        )

def test_application_exposes_defense_demo():
    source = APP.read_text(encoding="utf-8")

    required_markers = (
        '"Defense Demo"',
        "with tabs[8]:",
        'st.subheader("Defense Demo — Engineering Evaluation Workflow")',
        'st.markdown("### 1. Engineering problem and system")',
        'st.markdown("### 2. Candidate controller — IQL Seed 1")',
        'st.markdown("### 3. Engineering failure detection")',
        'st.markdown("### 4. Supervisory mitigation and measured trade-off")',
        'st.markdown("### 5. Engineering decision and acceptance")',
        'st.markdown("### 6. Verification and reproducibility")',
        "demo_iql_b1_comfort",
        "demo_guardrail_ready",
        "demo_interventions",
        'experiment_manifest["controller_artifacts"]',
        "preserved event records",
        "Validation boundary:",
    )

    for marker in required_markers:
        assert marker in source, f"Missing Defense Demo marker: {marker}"
