"""Tests for src/pipeline.py — CheckResult and TechnicalReviewRunner gate logic."""

import json

import pytest

from src.pipeline import (
    Category,
    CheckResult,
    Severity,
    TechnicalReviewRunner,
    check_solar,
    check_view_corridor,
)


def compliant_building(building_id="B-001", height_m=18.0, spacing_m=30.0):
    """Height 18 m -> required D = 28.8 m; spacing 30 m complies."""
    return {
        "properties": {
            "id": building_id,
            "building_type": "residential",
            "height_m": height_m,
            "spacing_to_south_m": spacing_m,
        }
    }


def violating_building(building_id="B-002", height_m=18.0, spacing_m=20.0):
    """Spacing 20 m < required 28.8 m: a confirmed solar violation."""
    return {
        "properties": {
            "id": building_id,
            "building_type": "residential",
            "height_m": height_m,
            "spacing_to_south_m": spacing_m,
        }
    }


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------


def test_check_result_creation_and_defaults():
    result = CheckResult(
        check_id="CHECK-SOLAR-001",
        check_name="Solar Access",
        passed=True,
        evidence={"total_residential": 2},
    )
    assert result.check_id == "CHECK-SOLAR-001"
    assert result.passed is True
    assert result.evidence == {"total_residential": 2}
    assert result.severity == Severity.MAJOR  # default severity
    assert result.category == Category.COMPUTE_INTENSIVE  # default category
    assert result.error is None


def test_check_result_to_dict_roundtrip():
    result = CheckResult(
        check_id="CHECK-X",
        check_name="X",
        passed=False,
        evidence={"k": 1},
        severity=Severity.CRITICAL,
        category=Category.DATA_INTENSIVE,
        detail="boom",
    )
    d = result.to_dict()
    assert d["check_id"] == "CHECK-X"
    assert d["passed"] is False
    assert d["severity"] == "critical"
    assert d["category"] == "B"
    assert d["detail"] == "boom"
    assert d["evidence"] == {"k": 1}


# ---------------------------------------------------------------------------
# check_solar / check_view_corridor behaviour
# ---------------------------------------------------------------------------


def test_check_solar_violation_fails_critical():
    result = check_solar([violating_building()])
    assert result.passed is False
    assert result.severity == Severity.CRITICAL
    assert result.evidence["violations"] == 1


def test_check_solar_missing_data_fails_closed():
    """A residential building without spacing_to_south_m cannot be verified
    and must fail closed (not silently pass)."""
    bldg = {"properties": {"id": "B-3", "building_type": "residential", "height_m": 18}}
    result = check_solar([bldg])
    assert result.passed is False
    assert result.severity == Severity.CRITICAL
    assert result.evidence["missing_spacing_data"] == 1


# ---------------------------------------------------------------------------
# TechnicalReviewRunner gate logic
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return TechnicalReviewRunner()


def test_runner_all_pass_gate_passes(runner):
    runner.register(check_solar)
    summary = runner.run_all(buildings=[compliant_building()])
    assert summary["gate_passed"] is True
    assert summary["total_checks"] == 1
    assert summary["passed"] == 1
    assert summary["blocking_failures"] == 0


def test_runner_critical_failure_blocks_gate(runner):
    runner.register(check_solar)
    summary = runner.run_all(buildings=[violating_building()])
    assert summary["gate_passed"] is False
    assert summary["failed"] == 1
    assert summary["blocking_failures"] == 1
    assert summary["results"][0]["severity"] == "critical"


def test_runner_not_assessed_does_not_block_gate(runner):
    """NOT_ASSESSED (data-gated) never blocks, but is reported honestly."""
    runner.register(check_view_corridor)
    summary = runner.run_all(buildings=[], corridors_defined=False)
    assert summary["gate_passed"] is True
    assert summary["not_assessed"] == 1
    assert summary["failed"] == 0
    assert len(summary["not_assessed_checks"]) == 1
    assert summary["not_assessed_checks"][0]["severity"] == "not_assessed"
    assert summary["not_assessed_checks"][0]["passed"] is False


def test_save_report_writes_json(tmp_path):
    """save_report persists the run_all report as a JSON file."""
    runner = TechnicalReviewRunner()
    runner.register(check_solar)
    out = tmp_path / "tech_review.json"
    report = runner.save_report(str(out), buildings=[compliant_building()])
    assert report["gate_passed"] is True
    with open(out, encoding="utf-8") as f:
        written = json.load(f)
    assert written == report


def test_runner_mixed_checks_gate_fail_with_not_assessed(runner):
    """Passing solar + failing solar + view corridor (NOT_ASSESSED):
    the gate fails on the CRITICAL only; NOT_ASSESSED stays separate.
    Also exercises signature filtering: run_all passes both ``buildings``
    and ``corridors_defined``, each check takes only what it declares."""
    runner.register(check_solar)
    runner.register(check_view_corridor)
    summary = runner.run_all(
        buildings=[compliant_building(), violating_building()],
        corridors_defined=False,
        land_use_features=[{"properties": {"area_sqm": 1000}}],  # declared by none
    )
    assert summary["gate_passed"] is False
    assert summary["blocking_failures"] == 1
    assert summary["not_assessed"] == 1
    assert summary["total_checks"] == 2
