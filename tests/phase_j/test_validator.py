"""Phase J validatorのSchema・V01〜V10相関回帰。"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest


pytest.importorskip("jsonschema")

from src.phase_j.contracts import ENUM_TYPES, OverlayHealth, OverlaySnapshot
from src.phase_j.validator import (
    PhaseJValidationError,
    ValidationContext,
    ValidationIssue,
    ValidationReport,
    make_fail_closed_snapshot,
    require_valid,
    validate_health,
    validate_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_EXAMPLES = ROOT / "docs/schemas/examples"
HEALTH_EXAMPLES = ROOT / "docs/schemas/health_examples"
PathParts = tuple[str | int, ...]
SNAPSHOT_REQUIRED_FIELDS = {
    (): ("schema_version", "identity", "timing", "display", "integrity", "mode", "evaluations", "input", "terminal", "runtime", "assets"),
    ("identity",): ("session_id", "stream_seq", "reducer_revision", "match_id", "match_state_seq", "input_generation"),
    ("timing",): ("source_available_frame", "source_available_ms", "published_at_utc", "capture_to_publish_latency_ms", "calculation_age_ms", "last_confirmed_age_ms"),
    ("display",): ("visibility", "status", "update_reason", "primary_hold_reason", "all_hold_reasons", "hold_started_ms", "hold_elapsed_ms"),
    ("integrity",): ("status", "fault_codes"),
    ("mode",): ("evaluation_mode", "practical_basis"),
    ("evaluations",): ("practical", "best_action", "player_adjusted"),
    ("evaluations", "practical"): ("availability", "request_id", "input_generation", "input_digest", "calculation_latency_ms", "calculation_age_ms", "p1_win_probability", "p2_win_probability", "advantage_score", "is_even", "origin", "calibration_id", "evaluated_positions"),
    ("evaluations", "best_action"): ("availability", "request_id", "input_generation", "input_digest", "calculation_latency_ms", "calculation_age_ms", "p1_position_value", "p1_position_value_low", "p1_position_value_high", "scale", "aggregation_profile_id", "search_profile_id", "searched_depth", "searched_nodes"),
    ("input",): ("event_seq", "p1_board_provenance", "p2_board_provenance", "recognition_quality", "unresolved_physics", "physical_prediction_used"),
    ("input", "recognition_quality"): ("status", "reason_codes"),
    ("terminal",): ("state", "winner", "evidence_kind", "result_code"),
    ("runtime",): ("status", "tier", "tier_profile_id", "practical_queue_depth", "best_action_queue_depth", "practical_worker_health", "best_action_worker_health", "discarded_jobs_since_match_start", "telemetry_health"),
    ("assets",): ("app_build_id", "recognition_model_hash", "recognition_config_hash", "prediction_model_hash", "calibration_hash"),
}
HEALTH_REQUIRED_FIELDS = (
    "schema_version", "server_status", "session_id", "latest_stream_seq",
    "latest_snapshot_age_ms", "subscriber_count", "subscriber_limit",
    "telemetry_health", "bound_host", "bound_port", "started_at_utc",
    "checked_at_utc", "last_error_code",
)
REQUIRED_CASES = tuple(
    ("snapshot", parent + (field_name,))
    for parent, fields in SNAPSHOT_REQUIRED_FIELDS.items()
    for field_name in fields
) + tuple(("health", (field_name,)) for field_name in HEALTH_REQUIRED_FIELDS)
ENUM_CASES = (
    ("visibility", "snapshot", "live_practical.json", ("display", "visibility")),
    ("display_status", "snapshot", "live_practical.json", ("display", "status")),
    ("update_reason", "snapshot", "live_practical.json", ("display", "update_reason")),
    ("hold_reason", "snapshot", "hold_visible.json", ("display", "primary_hold_reason")),
    ("hold_reason", "snapshot", "hold_visible.json", ("display", "all_hold_reasons", 0)),
    ("integrity_status", "snapshot", "live_practical.json", ("integrity", "status")),
    ("fault_code", "snapshot", "integrity_fault.json", ("integrity", "fault_codes", 0)),
    ("evaluation_mode", "snapshot", "live_practical.json", ("mode", "evaluation_mode")),
    ("evaluation_availability", "snapshot", "live_practical.json", ("evaluations", "practical", "availability")),
    ("evaluation_availability", "snapshot", "live_practical.json", ("evaluations", "best_action", "availability")),
    ("practical_origin", "snapshot", "live_practical.json", ("evaluations", "practical", "origin")),
    ("board_provenance", "snapshot", "live_practical.json", ("input", "p1_board_provenance")),
    ("board_provenance", "snapshot", "live_practical.json", ("input", "p2_board_provenance")),
    ("recognition_quality_status", "snapshot", "live_practical.json", ("input", "recognition_quality", "status")),
    ("recognition_reason", "snapshot", "hold_visible.json", ("input", "recognition_quality", "reason_codes", 0)),
    ("unresolved_physics_reason", "snapshot", "physical_prediction.json", ("input", "unresolved_physics", 0)),
    ("terminal_state", "snapshot", "terminal_p1.json", ("terminal", "state")),
    ("terminal_winner", "snapshot", "terminal_p1.json", ("terminal", "winner")),
    ("terminal_evidence_kind", "snapshot", "terminal_p1.json", ("terminal", "evidence_kind")),
    ("terminal_result_code", "snapshot", "terminal_p1.json", ("terminal", "result_code")),
    ("runtime_status", "snapshot", "live_practical.json", ("runtime", "status")),
    ("runtime_status", "health", "healthy.json", ("server_status",)),
    ("runtime_tier", "snapshot", "live_practical.json", ("runtime", "tier")),
    ("worker_health", "snapshot", "live_practical.json", ("runtime", "practical_worker_health")),
    ("worker_health", "snapshot", "live_practical.json", ("runtime", "best_action_worker_health")),
    ("telemetry_health", "snapshot", "live_practical.json", ("runtime", "telemetry_health")),
    ("telemetry_health", "health", "healthy.json", ("telemetry_health",)),
    ("health_error_code", "health", "degraded.json", ("last_error_code",)),
)


def _snapshot(name: str = "live_practical.json") -> dict[str, Any]:
    return json.loads((SNAPSHOT_EXAMPLES / name).read_text(encoding="utf-8"))


def _health(name: str = "healthy.json") -> dict[str, Any]:
    return json.loads((HEALTH_EXAMPLES / name).read_text(encoding="utf-8"))


def _assert_rule(report: ValidationReport, rule_id: str) -> None:
    assert not report.is_valid
    assert rule_id in {issue.rule_id for issue in report.issues}


def _set_unavailable(lane: dict[str, Any]) -> None:
    lane["availability"] = "unavailable"
    for key in lane:
        if key not in {"availability", "scale"}:
            lane[key] = None


def _parent_at(payload: dict[str, Any], path: PathParts) -> Any:
    parent: Any = payload
    for part in path[:-1]:
        parent = parent[part]
    return parent


def _path_id(path: PathParts) -> str:
    return ".".join(str(part) for part in path)


def _context() -> ValidationContext:
    return ValidationContext(
        allowed_calibration_ids=frozenset({"calibration-v1"}),
        calibration_score=lambda _calibration_id, probability: (probability - 0.5) * 200,
    )


def _mirror_practical(payload: dict[str, Any]) -> dict[str, Any]:
    mirrored = copy.deepcopy(payload)
    practical = mirrored["evaluations"]["practical"]
    practical["p1_win_probability"], practical["p2_win_probability"] = (
        practical["p2_win_probability"], practical["p1_win_probability"],
    )
    practical["advantage_score"] = -practical["advantage_score"]
    input_data = mirrored["input"]
    input_data["p1_board_provenance"], input_data["p2_board_provenance"] = (
        input_data["p2_board_provenance"], input_data["p1_board_provenance"],
    )
    return mirrored


def _mirror_terminal(payload: dict[str, Any]) -> dict[str, Any]:
    mirrored = _mirror_practical(payload)
    terminal = mirrored["terminal"]
    terminal["winner"] = "2P" if terminal["winner"] == "1P" else "1P"
    terminal["result_code"] = "p2_win" if terminal["result_code"] == "p1_win" else "p1_win"
    return mirrored


@pytest.mark.parametrize(
    ("kind", "path"),
    [pytest.param(kind, path, id=f"{kind}-{_path_id(path)}") for kind, path in REQUIRED_CASES],
)
def test_c02_each_required_field_is_rejected(kind: str, path: PathParts) -> None:
    payload = _snapshot() if kind == "snapshot" else _health()
    del _parent_at(payload, path)[path[-1]]
    report = validate_snapshot(payload) if kind == "snapshot" else validate_health(payload)
    assert not report.is_valid


def test_c03_case_table_covers_all_22_manifest_enums() -> None:
    covered = {enum_name for enum_name, _kind, _fixture, _path in ENUM_CASES}
    assert len(ENUM_TYPES) == 22
    assert covered == set(ENUM_TYPES)


@pytest.mark.parametrize(
    ("enum_name", "kind", "fixture", "path"),
    [
        pytest.param(name, kind, fixture, path, id=f"{name}-{kind}-{_path_id(path)}")
        for name, kind, fixture, path in ENUM_CASES
    ],
)
def test_c03_each_enum_schema_placement_rejects_unknown(
    enum_name: str,
    kind: str,
    fixture: str,
    path: PathParts,
) -> None:
    payload = _snapshot(fixture) if kind == "snapshot" else _health(fixture)
    _parent_at(payload, path)[path[-1]] = f"unknown-{enum_name}"
    report = validate_snapshot(payload) if kind == "snapshot" else validate_health(payload)
    assert not report.is_valid


def test_validation_value_objects_are_frozen_and_slotted() -> None:
    issue = ValidationIssue("R01", "$.x", "不正")
    report = ValidationReport((issue,))
    context = ValidationContext()
    assert not report.is_valid
    assert not hasattr(issue, "__dict__")
    with pytest.raises(FrozenInstanceError):
        context.even_threshold = 4.0  # type: ignore[misc]


def test_valid_mapping_and_overlay_snapshot_pass_without_mutation() -> None:
    payload = _snapshot()
    original = copy.deepcopy(payload)
    assert validate_snapshot(payload, _context()).is_valid
    assert validate_snapshot(OverlaySnapshot.from_mapping(payload), _context()).is_valid
    assert payload == original


def test_all_canonical_snapshot_and_health_examples_pass() -> None:
    snapshot_paths = sorted(SNAPSHOT_EXAMPLES.glob("*.json"))
    health_paths = sorted(HEALTH_EXAMPLES.glob("*.json"))
    assert len(snapshot_paths) == 8
    assert len(health_paths) == 5
    for path in snapshot_paths:
        assert validate_snapshot(json.loads(path.read_text(encoding="utf-8"))).is_valid, path.name
    for path in health_paths:
        assert validate_health(json.loads(path.read_text(encoding="utf-8"))).is_valid, path.name


def test_require_valid_returns_report_and_raises_with_issues() -> None:
    assert require_valid(_snapshot(), _context()).is_valid
    invalid = _snapshot()
    invalid["evaluations"]["practical"]["p2_win_probability"] = 0.5
    with pytest.raises(PhaseJValidationError) as caught:
        require_valid(invalid, _context())
    assert caught.value.report.issues


@pytest.mark.parametrize("invalid_kind", ["semantic", "schema"])
def test_make_fail_closed_snapshot_is_pure_legal_and_removes_all_values(
    invalid_kind: str,
) -> None:
    payload = _snapshot("hold_visible.json" if invalid_kind == "schema" else "live_practical.json")
    if invalid_kind == "semantic":
        payload["evaluations"]["practical"]["p2_win_probability"] = 0.5
    else:
        del payload["identity"]
    original = copy.deepcopy(payload)
    report = validate_snapshot(payload, _context())
    closed = make_fail_closed_snapshot(payload, report)
    result = closed.to_mapping()

    assert not report.is_valid
    assert payload == original
    assert result["display"] == {
        "visibility": "hidden", "status": "integrity_fault",
        "update_reason": "integrity_fault", "primary_hold_reason": None,
        "all_hold_reasons": [], "hold_started_ms": None, "hold_elapsed_ms": None,
    }
    assert result["integrity"]["fault_codes"] in (["schema_invalid"], ["internal_invariant_failed"])
    for lane_name in ("practical", "best_action"):
        lane = result["evaluations"][lane_name]
        assert lane["availability"] == "unavailable"
        assert all(value is None for key, value in lane.items() if key not in {"availability", "scale"})
    assert validate_snapshot(closed).is_valid


def test_make_fail_closed_snapshot_accepts_dto_without_mutating_it() -> None:
    payload = _snapshot()
    payload["evaluations"]["practical"]["p2_win_probability"] = 0.5
    dto = OverlaySnapshot.from_mapping(payload)
    before = dto.to_mapping()
    report = validate_snapshot(dto)
    closed = make_fail_closed_snapshot(dto, report)
    assert dto.to_mapping() == before
    assert validate_snapshot(closed).is_valid


def test_make_fail_closed_snapshot_handles_nonhashable_invalid_enums() -> None:
    payload = _snapshot()
    payload["mode"]["evaluation_mode"] = ["invalid"]
    payload["runtime"]["tier"] = {"invalid": True}
    report = validate_snapshot(payload)
    closed = make_fail_closed_snapshot(payload, report)
    result = closed.to_mapping()
    assert result["mode"]["evaluation_mode"] == "practical"
    assert result["runtime"]["tier"] == "standard"
    assert validate_snapshot(closed).is_valid


def test_module_imports_without_jsonschema_and_call_fails_explicitly() -> None:
    code = (
        "import src.phase_j.validator as v; "
        "v._jsonschema=None; v._schema_validator.cache_clear(); "
        "\ntry: v.validate_snapshot({})\nexcept RuntimeError: raise SystemExit(0)\n"
        "raise SystemExit(1)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("cause", ["hidden", "fault"])
def test_v01_hidden_or_fault_rejects_published_values(cause: str) -> None:
    payload = _snapshot()
    if cause == "hidden":
        payload["display"]["visibility"] = "hidden"
    else:
        payload["integrity"] = {"status": "fault", "fault_codes": ["sequence_gap"]}
    _assert_rule(validate_snapshot(payload), "S02")


@pytest.mark.parametrize("availability", ["pending", "unavailable"])
def test_v02_nonavailable_practical_rejects_evaluation_value(availability: str) -> None:
    payload = _snapshot()
    payload["evaluations"]["practical"]["availability"] = availability
    _assert_rule(validate_snapshot(payload), "S03")


def test_v03_probability_sum_outside_context_tolerance_is_rejected() -> None:
    payload = _snapshot()
    payload["evaluations"]["practical"]["p2_win_probability"] = 0.42001
    context = ValidationContext(probability_tolerance=1e-6)
    _assert_rule(validate_snapshot(payload, context), "R01")


@pytest.mark.parametrize("violation", ["score", "calibration_id"])
def test_v04_calibration_score_and_allowed_id_are_checked(violation: str) -> None:
    payload = _snapshot()
    if violation == "score":
        payload["evaluations"]["practical"]["advantage_score"] = 17
    else:
        payload["evaluations"]["practical"]["calibration_id"] = "unknown"
    _assert_rule(validate_snapshot(payload, _context()), "R02")


def test_v04_calibration_callback_failure_is_fail_closed() -> None:
    payload = _snapshot()

    def broken_callback(_calibration_id: str, _probability: float) -> float:
        raise RuntimeError("manifest unavailable")

    context = ValidationContext(calibration_score=broken_callback)
    _assert_rule(validate_snapshot(payload, context), "R02")


def test_v04_calibration_callback_non_numeric_result_is_fail_closed() -> None:
    payload = _snapshot()

    def non_numeric_callback(_calibration_id: str, _probability: float) -> Any:
        return "invalid"

    context = ValidationContext(calibration_score=non_numeric_callback)
    _assert_rule(validate_snapshot(payload, context), "R02")


def test_v05_even_flag_must_match_threshold() -> None:
    payload = _snapshot()
    payload["evaluations"]["practical"]["advantage_score"] = 3
    _assert_rule(validate_snapshot(payload), "R02")


@pytest.mark.parametrize("violation", ["generation", "age"])
def test_v06_live_generation_and_primary_age_must_match_root(violation: str) -> None:
    payload = _snapshot()
    if violation == "generation":
        payload["evaluations"]["practical"]["input_generation"] = 3
        rule = "R03"
    else:
        payload["timing"]["calculation_age_ms"] = 99
        rule = "R19"
    _assert_rule(validate_snapshot(payload), rule)


def test_v07_hold_allows_past_generation_but_rejects_future() -> None:
    payload = _snapshot("hold_visible.json")
    assert validate_snapshot(payload).is_valid
    payload["evaluations"]["practical"]["input_generation"] = 5
    _assert_rule(validate_snapshot(payload), "R03")


@pytest.mark.parametrize("violation", ["generation", "digest", "range"])
def test_v08_both_lane_mismatch_or_invalid_range_is_rejected_without_repair(
    violation: str,
) -> None:
    payload = _snapshot("both_lanes.json")
    if violation == "generation":
        payload["evaluations"]["best_action"]["input_generation"] = 3
        rule = "R04"
    elif violation == "digest":
        payload["evaluations"]["best_action"]["input_digest"] = "sha256:other"
        rule = "R04"
    else:
        payload["evaluations"]["best_action"]["p1_position_value_low"] = 20
        rule = "R20"
    original = copy.deepcopy(payload)
    _assert_rule(validate_snapshot(payload), rule)
    assert payload == original


@pytest.mark.parametrize("field", ["p1_win_probability", "result_code"])
def test_v09_terminal_direction_mismatch_is_rejected(field: str) -> None:
    payload = _snapshot("terminal_p1.json")
    if field == "p1_win_probability":
        payload["evaluations"]["practical"][field] = 0
    else:
        payload["terminal"][field] = "p2_win"
    _assert_rule(validate_snapshot(payload), "S05")


@pytest.mark.parametrize("violation", ["waiting_visible", "hold_metadata", "mode_lane"])
def test_v10_display_and_mode_forbidden_combinations_are_rejected(violation: str) -> None:
    payload = _snapshot()
    if violation == "waiting_visible":
        payload["display"]["status"] = "waiting"
    elif violation == "hold_metadata":
        payload["display"]["primary_hold_reason"] = "calculation_pending"
    else:
        payload["evaluations"]["best_action"] = _snapshot("both_lanes.json")["evaluations"]["best_action"]
    _assert_rule(validate_snapshot(payload), "R17" if violation != "mode_lane" else "R15")


@pytest.mark.parametrize("violation", ["healthy_telemetry", "content_fault"])
def test_v10_degraded_is_reserved_for_audit_unavailable(violation: str) -> None:
    payload = _snapshot()
    payload["integrity"]["status"] = "degraded"
    payload["runtime"]["status"] = "degraded"
    payload["runtime"]["telemetry_health"] = "unavailable"
    if violation == "healthy_telemetry":
        payload["runtime"]["telemetry_health"] = "healthy"
    else:
        payload["integrity"]["fault_codes"] = ["sequence_gap"]
    _assert_rule(validate_snapshot(payload), "R17")


def test_r11_physical_prediction_requires_projected_input() -> None:
    payload = _snapshot("physical_prediction.json")
    payload["input"]["physical_prediction_used"] = False
    _assert_rule(validate_snapshot(payload), "R11")


def test_r12_hold_start_cannot_be_after_source_time() -> None:
    payload = _snapshot("hold_visible.json")
    payload["display"]["hold_started_ms"] = 30501
    _assert_rule(validate_snapshot(payload), "R12")


def test_r16_available_result_requires_identifying_fields() -> None:
    payload = _snapshot()
    payload["evaluations"]["practical"]["request_id"] = None
    _assert_rule(validate_snapshot(payload), "R16")


def test_r16_terminal_fact_exempts_all_normal_request_metadata() -> None:
    payload = _snapshot("terminal_p1.json")
    metadata = (
        "request_id", "input_generation", "input_digest", "calculation_latency_ms",
        "calculation_age_ms", "calibration_id", "evaluated_positions",
    )
    for field_name in metadata:
        payload["evaluations"]["practical"][field_name] = None
    payload["timing"]["calculation_age_ms"] = None
    report = validate_snapshot(payload)
    assert report.is_valid, report.issues


@pytest.mark.parametrize("p1_probability", [0.0, 0.4, 0.49, 0.5, 0.7, 1.0])
def test_practical_probability_and_score_are_left_right_symmetric(
    p1_probability: float,
) -> None:
    payload = _snapshot()
    practical = payload["evaluations"]["practical"]
    practical["p1_win_probability"] = p1_probability
    practical["p2_win_probability"] = 1.0 - p1_probability
    practical["advantage_score"] = (p1_probability - 0.5) * 200
    practical["is_even"] = abs(practical["advantage_score"]) <= 3
    mirrored = _mirror_practical(payload)
    assert validate_snapshot(payload, _context()).is_valid
    assert validate_snapshot(mirrored, _context()).is_valid
    assert _mirror_practical(mirrored) == payload


def test_terminal_fact_is_left_right_symmetric_and_involutive() -> None:
    p1_terminal = _snapshot("terminal_p1.json")
    p2_terminal = _mirror_terminal(p1_terminal)
    assert p2_terminal["terminal"]["winner"] == "2P"
    assert p2_terminal["terminal"]["result_code"] == "p2_win"
    assert p2_terminal["evaluations"]["practical"]["p1_win_probability"] == 0
    assert p2_terminal["evaluations"]["practical"]["p2_win_probability"] == 1
    assert p2_terminal["evaluations"]["practical"]["advantage_score"] == -100
    assert validate_snapshot(p1_terminal).is_valid
    assert validate_snapshot(p2_terminal).is_valid
    assert _mirror_terminal(p2_terminal) == p1_terminal


def test_unknown_schema_structure_uses_generic_s01_rule() -> None:
    payload = _snapshot("initial_hidden.json")
    payload["assets"]["app_build_id"] = ""
    report = validate_snapshot(payload)
    _assert_rule(report, "S01")
    assert {issue.rule_id for issue in report.issues} == {"S01"}


def test_r18_restarting_worker_cannot_commit_available_result() -> None:
    payload = _snapshot()
    payload["runtime"]["practical_worker_health"] = "restarting"
    _assert_rule(validate_snapshot(payload), "R18")


@pytest.mark.parametrize(
    ("name", "mutation", "rule"),
    [
        ("healthy.json", ("last_error_code", "snapshot_stale"), "H01"),
        ("healthy.json", ("subscriber_count", 5), "H02"),
        ("healthy.json", ("latest_snapshot_age_ms", -1), "H03"),
        ("healthy.json", ("bound_host", "0.0.0.0"), "H04"),
    ],
)
def test_health_h01_to_h04_single_payload_rules(
    name: str, mutation: tuple[str, Any], rule: str,
) -> None:
    payload = _health(name)
    payload[mutation[0]] = mutation[1]
    _assert_rule(validate_health(payload), rule)


def test_health_valid_payload_and_report_require_valid() -> None:
    report = validate_health(_health())
    assert report.is_valid
    assert require_valid(report) is report


def test_overlay_health_dto_is_accepted() -> None:
    health = OverlayHealth.from_mapping(_health())
    assert validate_health(health).is_valid
