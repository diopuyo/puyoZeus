"""Phase J snapshot DTOの契約試験。"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from src.phase_j.contracts import (
    ENUM_MANIFEST_VERSION,
    ENUM_TYPES,
    HEALTH_SCHEMA_VERSION,
    NULLABLE_ENUM_NAMES,
    SNAPSHOT_SCHEMA_VERSION,
    OverlayHealth,
    OverlaySnapshot,
    deserialize_health,
    deserialize_snapshot,
    health_content_digest,
    serialize_health,
    serialize_snapshot,
    snapshot_content_digest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = PROJECT_ROOT / "docs" / "schemas"
EXAMPLE_DIR = SCHEMA_DIR / "examples"
HEALTH_EXAMPLE_DIR = SCHEMA_DIR / "health_examples"
SNAPSHOT_SCHEMA_PATH = SCHEMA_DIR / "puyo_overlay_snapshot_v1.schema.json"
HEALTH_SCHEMA_PATH = SCHEMA_DIR / "puyo_overlay_health_v1.schema.json"
ENUM_MANIFEST_PATH = SCHEMA_DIR / "puyo_overlay_enums_v1.json"
EXPECTED_CANONICAL_EXAMPLES = 8
EXPECTED_CANONICAL_HEALTH_EXAMPLES = 5

SchemaPath = tuple[str | int, ...]
SchemaLocation = tuple[str, SchemaPath]

ENUM_SCHEMA_LOCATIONS: dict[str, tuple[SchemaLocation, ...]] = {
    "visibility": (("snapshot", ("$defs", "display", "properties", "visibility", "enum")),),
    "display_status": (("snapshot", ("$defs", "display", "properties", "status", "enum")),),
    "update_reason": (("snapshot", ("$defs", "update_reason", "enum")),),
    "hold_reason": (("snapshot", ("$defs", "hold_reason", "enum")),),
    "integrity_status": (("snapshot", ("$defs", "integrity", "properties", "status", "enum")),),
    "fault_code": (("snapshot", ("$defs", "fault_code", "enum")),),
    "evaluation_mode": (("snapshot", ("$defs", "mode", "properties", "evaluation_mode", "enum")),),
    "evaluation_availability": (
        ("snapshot", ("$defs", "practical_evaluation", "properties", "availability", "enum")),
        ("snapshot", ("$defs", "best_action_evaluation", "properties", "availability", "enum")),
    ),
    "practical_origin": (("snapshot", ("$defs", "practical_evaluation", "properties", "origin", "enum")),),
    "board_provenance": (
        ("snapshot", ("$defs", "input", "properties", "p1_board_provenance", "enum")),
        ("snapshot", ("$defs", "input", "properties", "p2_board_provenance", "enum")),
    ),
    "recognition_quality_status": (
        ("snapshot", ("$defs", "input", "properties", "recognition_quality", "properties", "status", "enum")),
    ),
    "recognition_reason": (("snapshot", ("$defs", "recognition_reason", "enum")),),
    "unresolved_physics_reason": (("snapshot", ("$defs", "unresolved_physics_reason", "enum")),),
    "terminal_state": (("snapshot", ("$defs", "terminal", "properties", "state", "enum")),),
    "terminal_winner": (("snapshot", ("$defs", "terminal", "properties", "winner", "enum")),),
    "terminal_evidence_kind": (("snapshot", ("$defs", "terminal", "properties", "evidence_kind", "enum")),),
    "terminal_result_code": (("snapshot", ("$defs", "terminal", "properties", "result_code", "enum")),),
    "runtime_status": (
        ("snapshot", ("$defs", "runtime", "properties", "status", "enum")),
        ("health", ("properties", "server_status", "enum")),
    ),
    "runtime_tier": (("snapshot", ("$defs", "runtime", "properties", "tier", "enum")),),
    "worker_health": (
        ("snapshot", ("$defs", "runtime", "properties", "practical_worker_health", "enum")),
        ("snapshot", ("$defs", "runtime", "properties", "best_action_worker_health", "enum")),
    ),
    "telemetry_health": (
        ("snapshot", ("$defs", "runtime", "properties", "telemetry_health", "enum")),
        ("health", ("properties", "telemetry_health", "enum")),
    ),
    "health_error_code": (("health", ("properties", "last_error_code", "oneOf", 0, "enum")),),
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _read_schema_path(document: Any, path: SchemaPath) -> Any:
    current = document
    for part in path:
        current = current[part]
    return current


def test_c01_canonical_snapshot_examples_round_trip() -> None:
    paths = sorted(EXAMPLE_DIR.glob("*.json"))
    assert len(paths) == EXPECTED_CANONICAL_EXAMPLES
    for path in paths:
        payload = _load_json(path)
        snapshot = OverlaySnapshot.from_mapping(payload)
        assert snapshot.to_mapping() == payload, path.name
        assert deserialize_snapshot(serialize_snapshot(snapshot)) == snapshot, path.name


def test_c01_canonical_health_examples_round_trip() -> None:
    paths = sorted(HEALTH_EXAMPLE_DIR.glob("*.json"))
    assert len(paths) == EXPECTED_CANONICAL_HEALTH_EXAMPLES
    for path in paths:
        payload = _load_json(path)
        health = OverlayHealth.from_mapping(payload)
        assert health.to_mapping() == payload, path.name
        assert deserialize_health(serialize_health(health)) == health, path.name


def test_c02_missing_root_field_is_rejected() -> None:
    payload = _load_json(EXAMPLE_DIR / "initial_hidden.json")
    for root_field in OverlaySnapshot.ROOT_FIELDS:
        incomplete = dict(payload)
        incomplete.pop(root_field)
        with pytest.raises(ValueError, match=root_field):
            OverlaySnapshot.from_mapping(incomplete)


def test_c02_missing_health_root_field_is_rejected() -> None:
    payload = _load_json(HEALTH_EXAMPLE_DIR / "healthy.json")
    for root_field in OverlayHealth.ROOT_FIELDS:
        incomplete = dict(payload)
        incomplete.pop(root_field)
        with pytest.raises(ValueError, match=root_field):
            OverlayHealth.from_mapping(incomplete)


def test_c03_every_enum_rejects_unknown_value() -> None:
    assert len(ENUM_TYPES) == 22
    for enum_name, enum_type in ENUM_TYPES.items():
        with pytest.raises(ValueError, match="not-a-phase-j-enum-value"):
            enum_type("not-a-phase-j-enum-value")


def test_c04_unknown_root_field_is_immutable_and_round_trips() -> None:
    payload = _load_json(EXAMPLE_DIR / "initial_hidden.json")
    payload["future_extension"] = {"nested": [1, {"enabled": True}]}
    snapshot = OverlaySnapshot.from_mapping(payload)
    payload["future_extension"]["nested"][1]["enabled"] = False

    with pytest.raises(TypeError):
        snapshot.unknown_fields["future_extension"] = {}
    with pytest.raises(TypeError):
        snapshot.unknown_fields["future_extension"]["nested"][1]["enabled"] = False
    with pytest.raises(FrozenInstanceError):
        snapshot.schema_version = "changed"

    restored = snapshot.to_mapping()
    assert restored["future_extension"] == {"nested": [1, {"enabled": True}]}
    restored["future_extension"]["nested"][1]["enabled"] = False
    assert snapshot.to_mapping()["future_extension"]["nested"][1]["enabled"] is True


def test_c04_health_unknown_root_field_is_immutable_and_round_trips() -> None:
    payload = _load_json(HEALTH_EXAMPLE_DIR / "healthy.json")
    payload["future_health"] = {"samples": [1, 2]}
    health = OverlayHealth.from_mapping(payload)
    payload["future_health"]["samples"].append(3)
    with pytest.raises(TypeError):
        health.unknown_fields["future_health"]["samples"][0] = 9
    assert health.to_mapping()["future_health"] == {"samples": [1, 2]}


def test_c05_contract_enum_manifest_and_schema_match() -> None:
    manifest = _load_json(ENUM_MANIFEST_PATH)
    schemas = {
        "snapshot": _load_json(SNAPSHOT_SCHEMA_PATH),
        "health": _load_json(HEALTH_SCHEMA_PATH),
    }
    assert set(ENUM_TYPES) == set(manifest["enums"]) == set(ENUM_SCHEMA_LOCATIONS)
    for enum_name, enum_type in ENUM_TYPES.items():
        contract_values = {member.value for member in enum_type}
        manifest_values = set(manifest["enums"][enum_name]["values"])
        assert contract_values == manifest_values, enum_name
        for schema_name, path in ENUM_SCHEMA_LOCATIONS[enum_name]:
            schema_values = set(_read_schema_path(schemas[schema_name], path)) - {None}
            assert contract_values == schema_values, (enum_name, schema_name, path)
        nullable = bool(manifest["enums"][enum_name]["nullable"])
        assert nullable == (enum_name in NULLABLE_ENUM_NAMES), enum_name


def test_c05_schema_versions_match_contract_constants() -> None:
    snapshot_schema = _load_json(SNAPSHOT_SCHEMA_PATH)
    health_schema = _load_json(HEALTH_SCHEMA_PATH)
    manifest = _load_json(ENUM_MANIFEST_PATH)
    assert snapshot_schema["properties"]["schema_version"]["const"] == SNAPSHOT_SCHEMA_VERSION
    assert health_schema["properties"]["schema_version"]["const"] == HEALTH_SCHEMA_VERSION
    assert manifest["manifest_version"] == ENUM_MANIFEST_VERSION
    assert set(manifest["schema_versions"]) == {SNAPSHOT_SCHEMA_VERSION, HEALTH_SCHEMA_VERSION}


def test_c06_serialization_and_digest_are_deterministic() -> None:
    payload = _load_json(EXAMPLE_DIR / "both_lanes.json")
    reordered = {key: payload[key] for key in reversed(payload)}
    first = OverlaySnapshot.from_mapping(payload)
    second = OverlaySnapshot.from_mapping(reordered)
    assert serialize_snapshot(first) == serialize_snapshot(second)
    assert snapshot_content_digest(first) == snapshot_content_digest(second)
    assert snapshot_content_digest(first).startswith("sha256:")


def test_c06_health_serialization_and_digest_are_deterministic() -> None:
    payload = _load_json(HEALTH_EXAMPLE_DIR / "degraded.json")
    reordered = {key: payload[key] for key in reversed(payload)}
    first = OverlayHealth.from_mapping(payload)
    second = OverlayHealth.from_mapping(reordered)
    assert serialize_health(first) == serialize_health(second)
    assert health_content_digest(first) == health_content_digest(second)
    assert health_content_digest(first).startswith("sha256:")


def test_public_snapshot_does_not_expose_internal_analysis_fields() -> None:
    public_fields = set(OverlaySnapshot.__dataclass_fields__)
    assert {"board", "boards", "indicators", "debug"}.isdisjoint(public_fields)
