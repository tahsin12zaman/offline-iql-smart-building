"""Regression checks for the validated CityLearn IQL comfort-guardrail experiment."""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "citylearn_guarded_v2_summary.csv"
EVENTS = RESULTS / "citylearn_guarded_v2_events.csv"


def _summary():
    assert SUMMARY.exists(), f"Missing evidence artifact: {SUMMARY}"
    return pd.read_csv(SUMMARY)


def _events():
    assert EVENTS.exists(), f"Missing evidence artifact: {EVENTS}"
    return pd.read_csv(EVENTS)


def _rows():
    df = _summary()
    assert "mode" in df.columns
    assert {"NORMAL_IQL", "GUARDED_IQL"}.issubset(set(df["mode"]))
    normal = df[df["mode"] == "NORMAL_IQL"].iloc[0]
    guarded = df[df["mode"] == "GUARDED_IQL"].iloc[0]
    return normal, guarded


def test_same_719_step_evaluation_horizon():
    normal, guarded = _rows()
    assert int(normal["steps"]) == 719
    assert int(guarded["steps"]) == 719


def test_guardrail_reduces_overheating_in_every_building():
    normal, guarded = _rows()
    for building in (1, 2, 3):
        key = f"building_{building}_overheating_discomfort"
        assert float(guarded[key]) < float(normal[key])


def test_worst_building_comfort_result_matches_validated_experiment():
    normal, guarded = _rows()
    assert float(normal["worst_building_overheating_discomfort"]) * 100 == pytest.approx(97.76, abs=0.02)
    assert float(guarded["worst_building_overheating_discomfort"]) * 100 == pytest.approx(13.46, abs=0.02)


def test_resource_tradeoff_is_preserved():
    normal, guarded = _rows()
    for key in (
        "net_electricity_consumption",
        "electricity_cost",
        "carbon_emission",
        "peak_net_electricity",
    ):
        assert float(guarded[key]) > float(normal[key])


def test_event_count_matches_summary():
    _, guarded = _rows()
    events = _events()
    assert len(events) == 207
    assert int(guarded["guardrail_activations"]) == len(events)


def test_intervention_counts_by_building():
    events = _events()
    counts = events.groupby("building").size().to_dict()
    assert counts == {1: 96, 2: 67, 3: 44}


def test_runtime_state_uses_latest_completed_index():
    events = _events()
    required = {"building_time_step", "completed_index"}
    assert required.issubset(events.columns)
    assert (events["completed_index"] == events["building_time_step"] - 1).all()


def test_every_logged_intervention_changes_and_applies_fallback_cooling():
    events = _events()
    required = {
        "learned_cooling_action",
        "fallback_cooling_action",
        "applied_cooling_action",
        "action_difference",
    }
    assert required.issubset(events.columns)
    assert (events["action_difference"].abs() > 0).all()
    assert events["applied_cooling_action"].to_numpy() == pytest.approx(
        events["fallback_cooling_action"].to_numpy()
    )


def test_logged_events_satisfy_guard_condition():
    events = _events()
    required = {"occupant_count", "indoor_temperature", "upper_comfort_boundary"}
    assert required.issubset(events.columns)
    assert (events["occupant_count"] > 0).all()
    assert (events["indoor_temperature"] > events["upper_comfort_boundary"]).all()
