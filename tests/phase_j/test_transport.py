"""Phase J transport純粋層のT01〜T03契約試験。"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("jsonschema")

from src.phase_j.contracts import OverlayHealth, OverlaySnapshot
from src.phase_j.transport import (
    DEFAULT_HEARTBEAT_INTERVAL_MS,
    DEFAULT_SUBSCRIBER_LIMIT,
    DEFAULT_WRITE_TIMEOUT_MS,
    NO_STORE,
    ResponseSpec,
    SSE_CACHE_CONTROL,
    TransportPolicy,
    V1Route,
    build_events_response,
    build_health_response,
    build_latest_response,
    build_overlay_response,
    encode_heartbeat,
    encode_snapshot_event,
    require_loopback_bind,
    resolve_route,
)
from src.phase_j.validator import PhaseJValidationError

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "docs" / "schemas" / "examples"
HEALTH_EXAMPLES = ROOT / "docs" / "schemas" / "health_examples"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _snapshot(name: str = "initial_hidden.json") -> OverlaySnapshot:
    return OverlaySnapshot.from_mapping(_read_json(EXAMPLES / name))


def _health(name: str = "healthy.json") -> OverlayHealth:
    return OverlayHealth.from_mapping(_read_json(HEALTH_EXAMPLES / name))


def test_transport_policy_defaults_are_frozen_and_positive() -> None:
    policy = TransportPolicy()
    assert policy.subscriber_limit == DEFAULT_SUBSCRIBER_LIMIT == 4
    assert policy.write_timeout_ms == DEFAULT_WRITE_TIMEOUT_MS
    assert policy.heartbeat_interval_ms == DEFAULT_HEARTBEAT_INTERVAL_MS
    with pytest.raises(FrozenInstanceError):
        policy.subscriber_limit = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    "values",
    [
        {"status_code": "200"},
        {"content_type": ""},
        {"cache_control": "no-store\r\nInjected: 1"},
        {"body": "not-bytes"},
        {"streaming": 1},
    ],
)
def test_response_spec_rejects_invalid_field_types_and_header_injection(
    values: dict[str, Any],
) -> None:
    fields: dict[str, Any] = {
        "status_code": 200,
        "content_type": "text/plain",
        "cache_control": NO_STORE,
        "body": b"",
        "streaming": False,
    }
    fields.update(values)
    with pytest.raises((TypeError, ValueError)):
        ResponseSpec(**fields)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subscriber_limit", 0),
        ("subscriber_limit", True),
        ("write_timeout_ms", -1),
        ("write_timeout_ms", 1.5),
        ("heartbeat_interval_ms", 0),
        ("heartbeat_interval_ms", False),
    ],
)
def test_transport_policy_rejects_non_positive_integer_values(field: str, value: Any) -> None:
    values = {
        "subscriber_limit": DEFAULT_SUBSCRIBER_LIMIT,
        "write_timeout_ms": DEFAULT_WRITE_TIMEOUT_MS,
        "heartbeat_interval_ms": DEFAULT_HEARTBEAT_INTERVAL_MS,
    }
    values[field] = value
    with pytest.raises(ValueError, match=field):
        TransportPolicy(**values)


def test_t02_accepts_only_declared_ipv4_loopback() -> None:
    assert require_loopback_bind("127.0.0.1") == "127.0.0.1"


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "::1", "localhost", "example.test", "192.168.0.2", "127.0.0.2", "[::1]"],
)
def test_t02_rejects_wildcard_hostname_and_other_interfaces(host: str) -> None:
    with pytest.raises(ValueError, match="限定"):
        require_loopback_bind(host)


def test_t02_rejects_non_string_bind_without_coercion() -> None:
    with pytest.raises(TypeError):
        require_loopback_bind(123)  # type: ignore[arg-type]


def test_t01_resolves_only_exact_get_v1_routes() -> None:
    expected = {
        "/v1/overlay/": V1Route.OVERLAY,
        "/v1/overlay/events": V1Route.EVENTS,
        "/v1/overlay/latest": V1Route.LATEST,
        "/v1/overlay/health": V1Route.HEALTH,
    }
    assert {path: resolve_route("GET", path) for path in expected} == expected


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/v1/overlay/latest"),
        ("get", "/v1/overlay/latest"),
        ("HEAD", "/v1/overlay/health"),
        ("GET", "/v1/overlay/latest?cache=1"),
        ("GET", "/v1/overlay"),
        ("GET", "/v1/overlay/events/"),
        ("GET", "/"),
        ("GET", "/events"),
        ("GET", "/latest"),
        ("GET", "/unknown"),
    ],
)
def test_t01_unknown_method_path_and_query_do_not_resolve(method: str, path: str) -> None:
    assert resolve_route(method, path) is None


def test_t01_non_string_route_input_is_not_coerced() -> None:
    with pytest.raises(TypeError):
        resolve_route("GET", 1)  # type: ignore[arg-type]


def test_t01_response_headers_separate_cache_contracts() -> None:
    responses = {
        V1Route.OVERLAY: build_overlay_response(),
        V1Route.EVENTS: build_events_response(),
        V1Route.LATEST: build_latest_response(_snapshot()),
        V1Route.HEALTH: build_health_response(_health()),
    }
    for route in (V1Route.OVERLAY, V1Route.LATEST, V1Route.HEALTH):
        assert responses[route].headers["Cache-Control"] == NO_STORE
        assert "Content-Length" in responses[route].headers
    assert responses[V1Route.EVENTS].headers["Cache-Control"] == SSE_CACHE_CONTROL
    assert "Content-Length" not in responses[V1Route.EVENTS].headers
    assert responses[V1Route.EVENTS].content_type.startswith("text/event-stream")


def test_t01_v1_html_subscribes_only_to_v1_events() -> None:
    html = build_overlay_response().body.decode("utf-8")
    assert 'new EventSource("/v1/overlay/events")' in html
    assert 'new EventSource("/events")' not in html
    assert 'new EventSource("/latest")' not in html


def test_t01_latest_and_health_use_existing_canonical_serializers() -> None:
    snapshot = _snapshot()
    health = _health()
    latest = build_latest_response(snapshot)
    health_response = build_health_response(health)
    assert latest.body == snapshot.serialize().encode("utf-8")
    assert health_response.body == health.serialize().encode("utf-8")
    assert latest.content_type.startswith("application/json")
    assert health_response.content_type.startswith("application/json")


def test_t01_legacy_payload_objects_are_rejected() -> None:
    class AnalysisResultLike:
        pass

    legacy = AnalysisResultLike()
    with pytest.raises(TypeError, match="OverlaySnapshot"):
        build_latest_response(legacy)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="OverlayHealth"):
        build_health_response(legacy)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="OverlaySnapshot"):
        encode_snapshot_event(legacy)  # type: ignore[arg-type]


def test_t01_invalid_snapshot_and_health_are_explicitly_rejected() -> None:
    snapshot_payload = _snapshot().to_mapping()
    snapshot_payload["evaluations"]["practical"]["availability"] = "available"
    invalid_snapshot = OverlaySnapshot.from_mapping(snapshot_payload)
    with pytest.raises(PhaseJValidationError):
        build_latest_response(invalid_snapshot)

    health_payload = _health().to_mapping()
    health_payload["subscriber_count"] = health_payload["subscriber_limit"] + 1
    invalid_health = OverlayHealth.from_mapping(health_payload)
    with pytest.raises(PhaseJValidationError):
        build_health_response(invalid_health)


def test_t03_snapshot_event_is_canonical_single_line_sse() -> None:
    snapshot = _snapshot()
    encoded = encode_snapshot_event(snapshot)
    text = encoded.decode("utf-8")
    assert text == (
        "id: session-example:0\n"
        "event: snapshot\n"
        f"data: {snapshot.serialize()}\n\n"
    )
    assert text.splitlines()[:-1] == [
        "id: session-example:0",
        "event: snapshot",
        f"data: {snapshot.serialize()}",
    ]
    assert text.endswith("\n\n")


def test_t03_snapshot_event_rejects_session_id_header_injection() -> None:
    payload = _snapshot().to_mapping()
    payload["identity"]["session_id"] = "session\ninjected"
    with pytest.raises(ValueError, match="SSE改行"):
        encode_snapshot_event(OverlaySnapshot.from_mapping(payload))


def test_t03_heartbeat_is_comment_and_consumes_no_event_id() -> None:
    heartbeat = encode_heartbeat()
    assert heartbeat == b": heartbeat\n\n"
    assert b"id:" not in heartbeat
    assert b"event:" not in heartbeat
    assert b"data:" not in heartbeat
