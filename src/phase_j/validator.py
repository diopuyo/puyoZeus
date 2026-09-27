"""Phase J公開DTOのSchema・相関制約validator。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, TypeAlias

from .contracts import DisplayLayers, OverlayHealth, OverlaySnapshot, SNAPSHOT_SCHEMA_VERSION

try:  # jsonschema未導入でもmodule import自体は許す。
    import jsonschema as _jsonschema
except ModuleNotFoundError:  # pragma: no cover - 導入なし環境を別testで確認する。
    _jsonschema = None


CalibrationScore: TypeAlias = Callable[[str, float], float]
Payload: TypeAlias = Mapping[str, Any]
HealthPayload: TypeAlias = Payload | OverlayHealth
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "schemas"
SCHEMA_PATHS = {
    "snapshot": SCHEMA_DIR / "puyo_overlay_snapshot_v1.schema.json",
    "health": SCHEMA_DIR / "puyo_overlay_health_v1.schema.json",
}
RFC3339_DATETIME = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[Tt](\d{2}):(\d{2}):(\d{2})"
    r"(?:\.\d+)?(?:[Zz]|[+-](\d{2}):(\d{2}))$",
)
PRACTICAL_REQUIRED = (
    "request_id", "input_generation", "input_digest", "calculation_latency_ms",
    "calculation_age_ms", "p1_win_probability", "p2_win_probability",
    "advantage_score", "is_even", "origin", "calibration_id", "evaluated_positions",
)
BEST_ACTION_REQUIRED = (
    "request_id", "input_generation", "input_digest", "calculation_latency_ms",
    "calculation_age_ms", "p1_position_value", "p1_position_value_low",
    "p1_position_value_high", "aggregation_profile_id", "search_profile_id",
    "searched_depth", "searched_nodes",
)
TERMINAL_FACT_REQUIRED = frozenset(
    {"p1_win_probability", "p2_win_probability", "advantage_score", "is_even", "origin"},
)
EVALUATION_MODES = frozenset({"practical", "best_action", "both"})
RUNTIME_TIERS = frozenset({"lightweight", "standard", "high_accuracy", "analysis"})
ASSET_FIELDS = (
    "app_build_id", "recognition_model_hash", "recognition_config_hash",
    "prediction_model_hash", "calibration_hash",
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """一つの規則違反。pathはJSONPath風表記とする。"""

    rule_id: str
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """入力を変更しないvalidation結果。"""

    issues: tuple[ValidationIssue, ...]
    is_valid: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "is_valid", not self.issues)


@dataclass(frozen=True, slots=True)
class ValidationContext:
    """較正manifest依存値をvalidatorへ注入する。"""

    probability_tolerance: float = 1e-6
    even_threshold: float = 3.0
    calibration_score: CalibrationScore | None = None
    allowed_calibration_ids: frozenset[str] | None = None
    calibration_score_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.probability_tolerance < 0 or self.calibration_score_tolerance < 0:
            raise ValueError("許容誤差は非負である必要があります")
        if self.even_threshold < 0:
            raise ValueError("EVEN閾値は非負である必要があります")
        if self.allowed_calibration_ids is not None:
            object.__setattr__(
                self, "allowed_calibration_ids", frozenset(self.allowed_calibration_ids),
            )


class PhaseJValidationError(ValueError):
    """require_validが不正DTOを拒否したことを表す。"""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        summary = "; ".join(
            f"{issue.rule_id} {issue.path}: {issue.message}" for issue in report.issues
        )
        super().__init__(summary)


def _is_rfc3339_datetime(value: object) -> bool:
    """optional依存がなくてもRFC 3339日時をfail-openにしない。"""
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


def _json_path(parts: Any) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _issue(rule_id: str, path: str, message: str) -> ValidationIssue:
    return ValidationIssue(rule_id=rule_id, path=path, message=message)


def _report(issues: list[ValidationIssue]) -> ValidationReport:
    unique = {(item.rule_id, item.path, item.message): item for item in issues}
    ordered = tuple(unique[key] for key in sorted(unique))
    return ValidationReport(issues=ordered)


def _require_jsonschema() -> Any:
    if _jsonschema is None:
        raise RuntimeError(
            "Phase J validationにはjsonschema（Draft 2020-12対応）が必要です",
        )
    return _jsonschema


@lru_cache(maxsize=2)
def _schema_validator(kind: str) -> Any:
    module = _require_jsonschema()
    schema = json.loads(SCHEMA_PATHS[kind].read_text(encoding="utf-8"))
    validator_type = module.Draft202012Validator
    validator_type.check_schema(schema)
    checker = module.FormatChecker()
    checker.checks("date-time")(_is_rfc3339_datetime)
    return validator_type(schema, format_checker=checker)


def _schema_rule_snapshot(
    payload: Payload,
    path: tuple[Any, ...],
    schema_path: tuple[Any, ...],
    validator: str,
) -> str:
    display = payload.get("display", {})
    integrity = payload.get("integrity", {})
    status = display.get("status") if isinstance(display, Mapping) else None
    hidden = display.get("visibility") == "hidden" if isinstance(display, Mapping) else False
    fault = integrity.get("status") == "fault" if isinstance(integrity, Mapping) else False
    degraded = integrity.get("status") == "degraded" if isinstance(integrity, Mapping) else False
    terminal = payload.get("terminal", {})
    direction_bad = isinstance(terminal, Mapping) and (
        (terminal.get("winner") == "1P" and terminal.get("result_code") != "p1_win")
        or (terminal.get("winner") == "2P" and terminal.get("result_code") != "p2_win")
    )
    root = path[0] if path else None
    field = path[-1] if path else None
    conditional = "allOf" in schema_path
    if degraded and root in {"integrity", "runtime"}:
        return "R17"
    if root == "display" and field in {"hold_elapsed_ms", "hold_started_ms"}:
        return "R12"
    direction_fields = {"winner", "result_code", "p1_win_probability", "p2_win_probability", "advantage_score"}
    if status == "terminal_fact" and root == "terminal" and direction_bad:
        return "S05"
    if status == "terminal_fact" and field in direction_fields:
        return "S05"
    if status == "terminal_fact" and root in {"terminal", "evaluations"}:
        return "S04"
    if status == "result" and root in {"terminal", "evaluations"}:
        return "S06"
    if (hidden or fault) and root == "evaluations" and conditional:
        return "S02"
    if validator in {"enum", "const"}:
        return "S07"
    if root == "evaluations":
        return "S03"
    if root in {"integrity", "display"}:
        return "R17"
    return "S01"


def _schema_rule_health(
    payload: Payload,
    path: tuple[Any, ...],
    schema_path: tuple[Any, ...],
    validator: str,
) -> str:
    field = path[0] if path else None
    conditional = "allOf" in schema_path
    if validator in {"enum", "const"} and field != "bound_host" and not conditional:
        return "S07"
    if field in {"subscriber_count", "subscriber_limit"}:
        return "H02"
    if field == "latest_snapshot_age_ms":
        return "H03"
    if field in {"bound_host", "bound_port"}:
        return "H04"
    server, telemetry = payload.get("server_status"), payload.get("telemetry_health")
    last_error = payload.get("last_error_code")
    if server == "healthy" and (telemetry != "healthy" or last_error is not None):
        return "H01"
    if telemetry in {"degraded", "unavailable"} and server != "degraded":
        return "H01"
    if field in {"server_status", "telemetry_health", "last_error_code"}:
        return "H01"
    return "S01"


def _schema_issues(payload: Payload, kind: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    errors = sorted(
        _schema_validator(kind).iter_errors(payload),
        key=lambda error: (_json_path(error.absolute_path), error.message),
    )
    for error in errors:
        absolute_path = tuple(error.absolute_path)
        schema_path = tuple(error.absolute_schema_path)
        path = _json_path(absolute_path)
        rule = (
            _schema_rule_snapshot(payload, absolute_path, schema_path, error.validator)
            if kind == "snapshot"
            else _schema_rule_health(payload, absolute_path, schema_path, error.validator)
        )
        issues.append(_issue(rule, path, f"Schema違反: {error.message}"))
    return issues


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _snapshot_payload(value: Payload | OverlaySnapshot) -> dict[str, Any]:
    if isinstance(value, OverlaySnapshot):
        return value.to_mapping()
    if not isinstance(value, Mapping):
        raise TypeError("snapshotはmappingまたはOverlaySnapshotである必要があります")
    return _plain_json(value)


def _child_mapping(payload: Payload, field_name: str) -> Payload:
    value = payload.get(field_name, {})
    return value if isinstance(value, Mapping) else {}


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _nonempty_string(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) and value else fallback


def _fail_closed_identity(payload: Payload) -> dict[str, Any]:
    identity = _child_mapping(payload, "identity")
    match_id = identity.get("match_id")
    return {
        "session_id": _nonempty_string(identity.get("session_id"), "invalid-session"),
        "stream_seq": _nonnegative_int(identity.get("stream_seq")),
        "reducer_revision": _nonnegative_int(identity.get("reducer_revision")),
        "match_id": match_id if isinstance(match_id, str) or match_id is None else None,
        "match_state_seq": _nonnegative_int(identity.get("match_state_seq")),
        "input_generation": _nonnegative_int(identity.get("input_generation")),
    }


def _fail_closed_timing(payload: Payload) -> dict[str, Any]:
    timing = _child_mapping(payload, "timing")
    published = timing.get("published_at_utc")
    return {
        "source_available_frame": None,
        "source_available_ms": None,
        "published_at_utc": published if _is_rfc3339_datetime(published) else "1970-01-01T00:00:00Z",
        "capture_to_publish_latency_ms": None,
        "calculation_age_ms": None,
        "last_confirmed_age_ms": None,
    }


def _unavailable_practical() -> dict[str, Any]:
    return {
        "availability": "unavailable", "request_id": None,
        "input_generation": None, "input_digest": None,
        "calculation_latency_ms": None, "calculation_age_ms": None,
        "p1_win_probability": None, "p2_win_probability": None,
        "advantage_score": None, "is_even": None, "origin": None,
        "calibration_id": None, "evaluated_positions": None,
    }


def _unavailable_best_action() -> dict[str, Any]:
    return {
        "availability": "unavailable", "request_id": None,
        "input_generation": None, "input_digest": None,
        "calculation_latency_ms": None, "calculation_age_ms": None,
        "p1_position_value": None, "p1_position_value_low": None,
        "p1_position_value_high": None, "scale": "advantage_score",
        "aggregation_profile_id": None, "search_profile_id": None,
        "searched_depth": None, "searched_nodes": None,
    }


def _fail_closed_runtime(payload: Payload) -> dict[str, Any]:
    runtime = _child_mapping(payload, "runtime")
    tier = runtime.get("tier")
    tier = tier if isinstance(tier, str) and tier in RUNTIME_TIERS else "standard"
    return {
        "status": "degraded",
        "tier": tier,
        "tier_profile_id": _nonempty_string(runtime.get("tier_profile_id"), "fail-closed-v1"),
        "practical_queue_depth": 0,
        "best_action_queue_depth": 0,
        "practical_worker_health": "degraded",
        "best_action_worker_health": "degraded",
        "discarded_jobs_since_match_start": _nonnegative_int(runtime.get("discarded_jobs_since_match_start")),
        "telemetry_health": "healthy",
    }


def _fail_closed_assets(payload: Payload) -> dict[str, str]:
    assets = _child_mapping(payload, "assets")
    return {name: _nonempty_string(assets.get(name), "unavailable") for name in ASSET_FIELDS}


def _fail_closed_fault_code(report: ValidationReport) -> str:
    if any(issue.rule_id.startswith("S") for issue in report.issues):
        return "schema_invalid"
    return "internal_invariant_failed"


def make_fail_closed_snapshot(
    snapshot: Payload | OverlaySnapshot,
    report: ValidationReport,
) -> OverlaySnapshot:
    """不正snapshotを入力変更なしで合法な非表示snapshotへ変換する。"""
    payload = _snapshot_payload(snapshot)
    mode = _child_mapping(payload, "mode").get("evaluation_mode")
    mode = mode if isinstance(mode, str) and mode in EVALUATION_MODES else "practical"
    safe_payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "identity": _fail_closed_identity(payload),
        "timing": _fail_closed_timing(payload),
        "display": {
            "visibility": "hidden", "status": "integrity_fault",
            "update_reason": "integrity_fault", "primary_hold_reason": None,
            "all_hold_reasons": [], "hold_started_ms": None, "hold_elapsed_ms": None,
        },
        "integrity": {"status": "fault", "fault_codes": [_fail_closed_fault_code(report)]},
        "mode": {"evaluation_mode": mode, "practical_basis": "board_baseline"},
        "evaluations": {
            "practical": _unavailable_practical(),
            "best_action": _unavailable_best_action(),
            "player_adjusted": None,
        },
        "input": {
            "event_seq": None, "p1_board_provenance": "unknown",
            "p2_board_provenance": "unknown",
            "recognition_quality": {"status": "untrusted", "reason_codes": []},
            "unresolved_physics": [], "physical_prediction_used": False,
        },
        "terminal": {"state": "none", "winner": None, "evidence_kind": None, "result_code": None},
        "runtime": _fail_closed_runtime(payload),
        "assets": _fail_closed_assets(payload),
    }
    return OverlaySnapshot.from_mapping(safe_payload)


def _validate_probability_and_even(payload: Payload, context: ValidationContext) -> list[ValidationIssue]:
    practical = payload["evaluations"]["practical"]
    if practical["availability"] != "available":
        return []
    issues: list[ValidationIssue] = []
    p1, p2, score = practical["p1_win_probability"], practical["p2_win_probability"], practical["advantage_score"]
    if p1 is not None and p2 is not None and abs((p1 + p2) - 1.0) > context.probability_tolerance:
        issues.append(_issue("R01", "$.evaluations.practical", "P1/P2確率和が1ではありません"))
    if score is not None and practical["is_even"] is not None:
        expected_even = abs(score) <= context.even_threshold
        if practical["is_even"] != expected_even:
            issues.append(_issue("R02", "$.evaluations.practical.is_even", "EVEN閾値と一致しません"))
    issues.extend(_validate_calibration(payload, practical, context))
    return issues


def _validate_calibration(payload: Payload, practical: Payload, context: ValidationContext) -> list[ValidationIssue]:
    if payload["display"]["status"] == "terminal_fact":
        return []
    calibration_id = practical["calibration_id"]
    allowed = context.allowed_calibration_ids
    if allowed is not None and calibration_id not in allowed:
        return [_issue("R02", "$.evaluations.practical.calibration_id", "許可済み較正IDではありません")]
    if context.calibration_score is None or not isinstance(calibration_id, str):
        return []
    try:
        expected = context.calibration_score(calibration_id, practical["p1_win_probability"])
    except Exception as exc:  # callback障害を成功扱いしない。
        return [_issue("R02", "$.evaluations.practical.calibration_id", f"較正照合に失敗しました: {exc}")]
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return [_issue("R02", "$.evaluations.practical.advantage_score", "較正照合結果が数値ではありません")]
    if abs(expected - practical["advantage_score"]) <= context.calibration_score_tolerance:
        return []
    return [_issue("R02", "$.evaluations.practical.advantage_score", "較正済み確率とscoreが一致しません")]


def _available_lanes(payload: Payload) -> list[tuple[str, Payload]]:
    evaluations = payload["evaluations"]
    return [
        (name, evaluations[name])
        for name in ("practical", "best_action")
        if evaluations[name]["availability"] == "available"
    ]


def _validate_generations(payload: Payload) -> list[ValidationIssue]:
    root_generation = payload["identity"]["input_generation"]
    status = payload["display"]["status"]
    issues: list[ValidationIssue] = []
    for name, lane in _available_lanes(payload):
        generation = lane["input_generation"]
        if generation is None:
            continue
        path = f"$.evaluations.{name}.input_generation"
        if generation > root_generation:
            issues.append(_issue("R03", path, "lane generationがrootより未来です"))
        elif status in {"live", "physical_prediction"} and generation != root_generation:
            issues.append(_issue("R03", path, "live表示のgenerationがrootと一致しません"))
    return issues


def _validate_both(payload: Payload) -> list[ValidationIssue]:
    if payload["mode"]["evaluation_mode"] != "both":
        return []
    practical = payload["evaluations"]["practical"]
    best = payload["evaluations"]["best_action"]
    if practical["availability"] != "available" or best["availability"] != "available":
        return []
    issues: list[ValidationIssue] = []
    if practical["input_generation"] != best["input_generation"]:
        issues.append(_issue("R04", "$.evaluations", "both laneのgenerationが一致しません"))
    if practical["input_digest"] != best["input_digest"]:
        issues.append(_issue("R04", "$.evaluations", "both laneのdigestが一致しません"))
    return issues


def _validate_provenance(payload: Payload) -> list[ValidationIssue]:
    status = payload["display"]["status"]
    input_data = payload["input"]
    provenances = (input_data["p1_board_provenance"], input_data["p2_board_provenance"])
    issues: list[ValidationIssue] = []
    if status == "live" and provenances != ("confirmed", "confirmed"):
        issues.append(_issue("R11", "$.input", "live表示には両者confirmedが必要です"))
    if status == "physical_prediction":
        if "physics_projected" not in provenances or not input_data["physical_prediction_used"]:
            issues.append(_issue("R11", "$.input", "物理予測のprovenanceが不足しています"))
        practical = payload["evaluations"]["practical"]
        if practical["availability"] == "available" and practical["origin"] != "physical_prediction":
            issues.append(_issue("R11", "$.evaluations.practical.origin", "物理予測originではありません"))
    return issues


def _validate_mode(payload: Payload) -> list[ValidationIssue]:
    status = payload["display"]["status"]
    if status in {"terminal_fact", "result"}:
        return []
    mode = payload["mode"]["evaluation_mode"]
    evaluations = payload["evaluations"]
    hidden_lane = "best_action" if mode == "practical" else "practical" if mode == "best_action" else None
    if hidden_lane is None or evaluations[hidden_lane]["availability"] == "unavailable":
        return []
    return [_issue("R15", f"$.evaluations.{hidden_lane}.availability", "非選択laneがunavailableではありません")]


def _validate_available_fields(payload: Payload) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    terminal = payload["display"]["status"] == "terminal_fact"
    for name, lane in _available_lanes(payload):
        required = PRACTICAL_REQUIRED if name == "practical" else BEST_ACTION_REQUIRED
        if terminal and name == "practical":
            required = tuple(field for field in required if field in TERMINAL_FACT_REQUIRED)
        missing = [field for field in required if lane[field] is None]
        for field_name in missing:
            issues.append(_issue("R16", f"$.evaluations.{name}.{field_name}", "available結果の必須値がnullです"))
    return issues


def _validate_hold_time(payload: Payload) -> list[ValidationIssue]:
    display = payload["display"]
    if display["status"] != "hold":
        return []
    source_ms = payload["timing"]["source_available_ms"]
    started_ms = display["hold_started_ms"]
    if source_ms is not None and started_ms is not None and started_ms > source_ms:
        return [_issue("R12", "$.display.hold_started_ms", "hold開始映像時刻が現在入力より未来です")]
    return []


def _validate_display_integrity(payload: Payload) -> list[ValidationIssue]:
    display, integrity = payload["display"], payload["integrity"]
    status, visibility = display["status"], display["visibility"]
    faults, telemetry = integrity["fault_codes"], payload["runtime"]["telemetry_health"]
    issues: list[ValidationIssue] = []
    visible_statuses = {"live", "physical_prediction", "hold", "terminal_fact", "result"}
    if status in visible_statuses and visibility != "visible":
        issues.append(_issue("R17", "$.display.visibility", "表示状態にはvisibleが必要です"))
    if status in {"waiting", "integrity_fault"} and visibility != "hidden":
        issues.append(_issue("R17", "$.display.visibility", "待機/fault状態はhiddenです"))
    if status != "hold" and _has_hold_metadata(display):
        issues.append(_issue("R17", "$.display", "hold以外にhold metadataが残っています"))
    if integrity["status"] == "fault" and (status != "integrity_fault" or not faults):
        issues.append(_issue("R17", "$.integrity", "fault状態と表示またはfault codeが不整合です"))
    if integrity["status"] != "fault" and faults:
        issues.append(_issue("R17", "$.integrity.fault_codes", "非fault状態にfault codeがあります"))
    if telemetry in {"degraded", "unavailable"} and integrity["status"] == "ok":
        issues.append(_issue("R17", "$.integrity.status", "監査不能時はdegradedが必要です"))
    return issues


def _has_hold_metadata(display: Payload) -> bool:
    return any(
        (display["primary_hold_reason"] is not None, bool(display["all_hold_reasons"]),
         display["hold_started_ms"] is not None, display["hold_elapsed_ms"] is not None),
    )


def _validate_worker_lanes(payload: Payload) -> list[ValidationIssue]:
    runtime, evaluations = payload["runtime"], payload["evaluations"]
    issues: list[ValidationIssue] = []
    for name in ("practical", "best_action"):
        health = runtime[f"{name}_worker_health"]
        availability = evaluations[name]["availability"]
        queue_depth = runtime[f"{name}_queue_depth"]
        path = f"$.runtime.{name}_worker_health"
        if health == "disabled" and (availability != "unavailable" or queue_depth != 0):
            issues.append(_issue("R18", path, "disabled workerのlaneまたはqueueが公開されています"))
        if health in {"starting", "restarting"} and availability == "available":
            issues.append(_issue("R18", path, "準備中workerの結果はavailableにできません"))
        if health == "degraded" and availability == "available" and payload["display"]["status"] != "hold":
            issues.append(_issue("R18", path, "degraded workerの既存値はholdでのみ保持できます"))
    return issues


def _validate_root_age(payload: Payload, context: ValidationContext) -> list[ValidationIssue]:
    evaluations = payload["evaluations"]
    practical, best = evaluations["practical"], evaluations["best_action"]
    expected = None
    if practical["availability"] == "available":
        expected = practical["calculation_age_ms"]
    elif payload["mode"]["evaluation_mode"] == "best_action" and best["availability"] == "available":
        expected = best["calculation_age_ms"]
    actual = payload["timing"]["calculation_age_ms"]
    if expected is None and actual is None:
        return []
    if expected is not None and actual is not None and expected == actual:
        return []
    return [_issue("R19", "$.timing.calculation_age_ms", "表示主laneのageと一致しません")]


def _validate_best_action(payload: Payload) -> list[ValidationIssue]:
    best = payload["evaluations"]["best_action"]
    if best["availability"] != "available":
        return []
    low, central, high = (
        best["p1_position_value_low"], best["p1_position_value"], best["p1_position_value_high"],
    )
    if None not in {low, central, high} and low <= central <= high:
        return []
    return [_issue("R20", "$.evaluations.best_action", "low <= central <= highを満たしません")]


def _snapshot_semantic_issues(payload: Payload, context: ValidationContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for validator in (
        _validate_probability_and_even, _validate_root_age,
    ):
        issues.extend(validator(payload, context))
    for validator in (
        _validate_generations, _validate_both, _validate_provenance, _validate_mode,
        _validate_available_fields, _validate_hold_time, _validate_display_integrity,
        _validate_worker_lanes, _validate_best_action, _validate_layers,
    ):
        issues.extend(validator(payload))
    return issues


def _validate_layers(payload: Payload) -> list[ValidationIssue]:
    """旧DTOは受理し、C案が載るときは欠落・範囲・表示値相関を検査する。"""
    layers = payload['evaluations'].get('display_layers')
    if layers is None:
        return []
    try:
        parsed = DisplayLayers(**layers)
        practical = payload['evaluations']['practical']
        expected = practical['p1_win_probability'] if practical['availability'] == 'available' else None
        if parsed.displayed_p1 != expected:
            raise ValueError('層別表示値と実配信値が一致しません')
    except (ValueError, TypeError) as error:
        return [_issue('R21', '$.evaluations.display_layers', str(error))]
    return []


def validate_snapshot(
    snapshot: Payload | OverlaySnapshot,
    context: ValidationContext | None = None,
) -> ValidationReport:
    """snapshot単体で判定可能なSchema/R01〜R20制約を検査する。"""
    payload = _snapshot_payload(snapshot)
    schema_issues = _schema_issues(payload, "snapshot")
    if schema_issues:
        return _report(schema_issues)
    return _report(_snapshot_semantic_issues(payload, context or ValidationContext()))


def _health_semantic_issues(payload: Payload) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if payload["subscriber_count"] > payload["subscriber_limit"]:
        issues.append(_issue("H02", "$.subscriber_count", "subscriber上限を超えています"))
    return issues


def validate_health(health: HealthPayload) -> ValidationReport:
    """health単体で判定可能なSchema/H01〜H04制約を検査する。"""
    if isinstance(health, OverlayHealth):
        health = health.to_mapping()
    if not isinstance(health, Mapping):
        raise TypeError("healthはmappingまたはOverlayHealthである必要があります")
    payload = _plain_json(health)
    schema_issues = _schema_issues(payload, "health")
    if schema_issues:
        return _report(schema_issues)
    return _report(_health_semantic_issues(payload))


def require_valid(
    value: Payload | OverlaySnapshot | ValidationReport,
    context: ValidationContext | None = None,
) -> ValidationReport:
    """不正snapshot/reportを例外化し、正常reportを返す。"""
    report = value if isinstance(value, ValidationReport) else validate_snapshot(value, context)
    if not report.is_valid:
        raise PhaseJValidationError(report)
    return report


__all__ = [
    "PhaseJValidationError",
    "ValidationContext",
    "ValidationIssue",
    "ValidationReport",
    "make_fail_closed_snapshot",
    "require_valid",
    "validate_health",
    "validate_snapshot",
]
