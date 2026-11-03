"""Phase J snapshot SchemaのP1契約回帰。"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest


jsonschema = pytest.importorskip("jsonschema")
Draft202012Validator = jsonschema.Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "docs/schemas/puyo_overlay_snapshot_v1.schema.json"
HEALTH_SCHEMA_PATH = ROOT / "docs/schemas/puyo_overlay_health_v1.schema.json"
EXAMPLE_DIR = ROOT / "docs/schemas/examples"
HEALTH_EXAMPLE_DIR = ROOT / "docs/schemas/health_examples"
IDENTIFIER_VALUES: dict[str, Any] = {
    "request_id": "stale-request",
    "input_generation": 4,
    "input_digest": "sha256:stale",
}
PRACTICAL_RESULT_FIELDS = (
    "calculation_latency_ms", "calculation_age_ms", "p1_win_probability",
    "p2_win_probability", "advantage_score", "is_even", "origin",
    "calibration_id", "evaluated_positions",
)
BEST_ACTION_RESULT_FIELDS = (
    "calculation_latency_ms", "calculation_age_ms", "p1_position_value",
    "p1_position_value_low", "p1_position_value_high", "aggregation_profile_id",
    "search_profile_id", "searched_depth", "searched_nodes",
)
RFC3339_DATETIME = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[Tt](\d{2}):(\d{2}):(\d{2})"
    r"(?:\.\d+)?(?:[Zz]|[+-](\d{2}):(\d{2}))$",
)


def _is_rfc3339_datetime(value: object) -> bool:
    """optional依存なしでもRFC 3339の日時をfail-openにしない。"""
    if not isinstance(value, str):
        return False
    matched = RFC3339_DATETIME.fullmatch(value)
    if matched is None:
        return False
    year, month, day, hour, minute, second, zone_hour, zone_minute = matched.groups()
    try:
        datetime(int(year), int(month), int(day), int(hour), int(minute), min(int(second), 59))
    except ValueError:
        return False
    return int(second) <= 60 and (
        zone_hour is None or (int(zone_hour) <= 23 and int(zone_minute) <= 59)
    )


FORMAT_CHECKER = jsonschema.FormatChecker()
FORMAT_CHECKER.checks("date-time")(_is_rfc3339_datetime)


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    """format assertionを有効にしたsnapshot validatorを返す。"""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema, format_checker=FORMAT_CHECKER,
    )


@pytest.fixture(scope="module")
def health_validator() -> Draft202012Validator:
    """format assertionを有効にしたhealth validatorを返す。"""
    schema = json.loads(HEALTH_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema, format_checker=FORMAT_CHECKER,
    )


def _example(name: str) -> dict[str, Any]:
    return json.loads((EXAMPLE_DIR / name).read_text(encoding="utf-8"))


def _health_example(name: str) -> dict[str, Any]:
    return json.loads((HEALTH_EXAMPLE_DIR / name).read_text(encoding="utf-8"))


def _set_lane_without_result(
    snapshot: dict[str, Any], lane: str, availability: str,
) -> dict[str, Any]:
    result = copy.deepcopy(snapshot)
    evaluation = result["evaluations"][lane]
    evaluation["availability"] = availability
    fields = PRACTICAL_RESULT_FIELDS if lane == "practical" else BEST_ACTION_RESULT_FIELDS
    for field in fields:
        evaluation[field] = None
    return result


def _visible_non_display_lane(lane: str) -> dict[str, Any]:
    """hidden/faultの補助定義に頼らずunavailableを検査する。"""
    name = "both_lanes.json" if lane == "practical" else "live_practical.json"
    result = _set_lane_without_result(_example(name), lane, "unavailable")
    result["mode"]["evaluation_mode"] = "best_action" if lane == "practical" else "practical"
    for field in IDENTIFIER_VALUES:
        result["evaluations"][lane][field] = None
    return result


def _expired_hold(lane: str = "practical") -> dict[str, Any]:
    snapshot = _set_lane_without_result(
        _example("hold_visible.json"), lane, "pending",
    )
    snapshot["display"]["update_reason"] = "hold_expired"
    return snapshot


def _both_mode_hold() -> dict[str, Any]:
    snapshot = _example("both_lanes.json")
    snapshot["display"] = copy.deepcopy(_example("hold_visible.json")["display"])
    snapshot["display"]["update_reason"] = "hold_expired"
    return snapshot


@pytest.mark.parametrize("lane", ["practical", "best_action"])
@pytest.mark.parametrize("identifier", list(IDENTIFIER_VALUES))
def test_v02_unavailable_lane_rejects_each_stale_identifier(
    validator: Draft202012Validator, lane: str, identifier: str,
) -> None:
    snapshot = _visible_non_display_lane(lane)
    snapshot["evaluations"][lane][identifier] = IDENTIFIER_VALUES[identifier]
    assert list(validator.iter_errors(snapshot))


@pytest.mark.parametrize("lane", ["practical", "best_action"])
def test_v02_pending_lane_may_retain_job_identifiers(
    validator: Draft202012Validator, lane: str,
) -> None:
    snapshot = _visible_non_display_lane(lane)
    snapshot["evaluations"][lane]["availability"] = "pending"
    snapshot["evaluations"][lane].update(IDENTIFIER_VALUES)
    assert list(validator.iter_errors(snapshot)) == []


def test_c01_all_eight_canonical_snapshots_pass_schema(
    validator: Draft202012Validator,
) -> None:
    examples = sorted(EXAMPLE_DIR.glob("*.json"))
    assert len(examples) == 8
    for path in examples:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert list(validator.iter_errors(payload)) == [], path.name


def test_c01_all_five_canonical_health_payloads_pass_schema(
    health_validator: Draft202012Validator,
) -> None:
    examples = sorted(HEALTH_EXAMPLE_DIR.glob("*.json"))
    assert len(examples) == 5
    for path in examples:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert list(health_validator.iter_errors(payload)) == [], path.name


@pytest.mark.parametrize("path", [SCHEMA_PATH, HEALTH_SCHEMA_PATH])
def test_c01_schema_document_is_valid_draft_2020_12(path: Path) -> None:
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


def test_c01_snapshot_and_health_reject_invalid_rfc3339_datetime(
    validator: Draft202012Validator,
    health_validator: Draft202012Validator,
) -> None:
    snapshot = _example("live_practical.json")
    snapshot["timing"]["published_at_utc"] = "not-rfc3339"
    health = _health_example("healthy.json")
    health["checked_at_utc"] = "not-rfc3339"
    assert list(validator.iter_errors(snapshot))
    assert list(health_validator.iter_errors(health))


def test_r06_expired_hold_is_visible_with_pending_target(
    validator: Draft202012Validator,
) -> None:
    snapshot = _expired_hold()
    assert list(validator.iter_errors(snapshot)) == []


@pytest.mark.parametrize(
    ("field", "value"), [("status", "live"), ("visibility", "hidden")],
)
def test_v10_expired_hold_rejects_wrong_display_state(
    validator: Draft202012Validator, field: str, value: str,
) -> None:
    snapshot = _expired_hold()
    snapshot["display"][field] = value
    assert list(validator.iter_errors(snapshot))


def test_v10_expired_hold_rejects_available_target(
    validator: Draft202012Validator,
) -> None:
    expired = _example("hold_visible.json")
    expired["display"]["update_reason"] = "hold_expired"
    assert list(validator.iter_errors(expired))


def test_r06_both_mode_expired_hold_allows_one_pending_lane(
    validator: Draft202012Validator,
) -> None:
    snapshot = _set_lane_without_result(_both_mode_hold(), "best_action", "pending")
    assert list(validator.iter_errors(snapshot)) == []


def test_r06_both_mode_expired_hold_rejects_both_available(
    validator: Draft202012Validator,
) -> None:
    assert list(validator.iter_errors(_both_mode_hold()))


def test_v10_hold_requires_hold_started_ms(
    validator: Draft202012Validator,
) -> None:
    snapshot = _example("hold_visible.json")
    snapshot["display"]["hold_started_ms"] = None
    assert list(validator.iter_errors(snapshot))


def test_r06_hold_elapsed_ms_accepts_nonnegative_process_age(
    validator: Draft202012Validator,
) -> None:
    snapshot = _example("hold_visible.json")
    snapshot["display"]["hold_elapsed_ms"] = 1000
    assert list(validator.iter_errors(snapshot)) == []


@pytest.mark.parametrize("value", [None, -1])
def test_v10_hold_rejects_invalid_hold_elapsed_ms(
    validator: Draft202012Validator, value: int | None,
) -> None:
    snapshot = _example("hold_visible.json")
    snapshot["display"]["hold_elapsed_ms"] = value
    assert list(validator.iter_errors(snapshot))


def test_v10_non_hold_requires_null_hold_elapsed_ms(
    validator: Draft202012Validator,
) -> None:
    snapshot = _example("live_practical.json")
    snapshot["display"]["hold_elapsed_ms"] = 0
    assert list(validator.iter_errors(snapshot))


def test_v10_allows_degraded_integrity_only_when_audit_is_unavailable(
    validator: Draft202012Validator,
) -> None:
    snapshot = _example("live_practical.json")
    snapshot["integrity"]["status"] = "degraded"
    snapshot["runtime"]["status"] = "degraded"
    snapshot["runtime"]["telemetry_health"] = "unavailable"
    assert list(validator.iter_errors(snapshot)) == []


@pytest.mark.parametrize("violation", ["healthy_telemetry", "content_fault"])
def test_v10_rejects_degraded_integrity_outside_audit_failure(
    validator: Draft202012Validator, violation: str,
) -> None:
    snapshot = _example("live_practical.json")
    snapshot["integrity"]["status"] = "degraded"
    snapshot["runtime"]["status"] = "degraded"
    snapshot["runtime"]["telemetry_health"] = "unavailable"
    if violation == "healthy_telemetry":
        snapshot["runtime"]["telemetry_health"] = "healthy"
    else:
        snapshot["integrity"]["fault_codes"] = ["sequence_gap"]
    assert list(validator.iter_errors(snapshot))
