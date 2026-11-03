"""Phase J v1 HTTP/SSE transportの純粋な契約とencoding。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from .contracts import OverlayHealth, OverlaySnapshot, serialize_health, serialize_snapshot
from .validator import PhaseJValidationError, validate_health, validate_snapshot

DEFAULT_SUBSCRIBER_LIMIT = 4
DEFAULT_WRITE_TIMEOUT_MS = 5_000
DEFAULT_HEARTBEAT_INTERVAL_MS = 15_000
HTTP_OK = 200
HTML_CONTENT_TYPE = "text/html; charset=utf-8"
JSON_CONTENT_TYPE = "application/json; charset=utf-8"
SSE_CONTENT_TYPE = "text/event-stream"
NO_STORE = "no-store"
SSE_CACHE_CONTROL = "no-cache, no-transform"
ALLOWED_LOOPBACK_BINDS = frozenset({"127.0.0.1"})

V1_OVERLAY_HTML = """<!doctype html>
<html lang="ja">
<head><meta charset="utf-8"><title>Puyo Overlay v1</title></head>
<body><main id="puyo-overlay" aria-live="polite"></main>
<script>
const events = new EventSource("/v1/overlay/events");
events.addEventListener("snapshot", event => {
  window.dispatchEvent(new CustomEvent("puyo-overlay-snapshot", {detail: JSON.parse(event.data)}));
});
</script></body></html>
"""


class V1Route(StrEnum):
    """legacy endpointと混在しないPhase J v1 route。"""

    OVERLAY = "/v1/overlay/"
    EVENTS = "/v1/overlay/events"
    LATEST = "/v1/overlay/latest"
    HEALTH = "/v1/overlay/health"


@dataclass(frozen=True, slots=True)
class TransportPolicy:
    """live serverが後段で適用する上限・timeout契約。"""

    subscriber_limit: int = DEFAULT_SUBSCRIBER_LIMIT
    write_timeout_ms: int = DEFAULT_WRITE_TIMEOUT_MS
    heartbeat_interval_ms: int = DEFAULT_HEARTBEAT_INTERVAL_MS

    def __post_init__(self) -> None:
        for name in ("subscriber_limit", "write_timeout_ms", "heartbeat_interval_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name}は正の整数である必要があります")


@dataclass(frozen=True, slots=True)
class ResponseSpec:
    """socketへ依存しないHTTP response契約。"""

    status_code: int
    content_type: str
    cache_control: str
    body: bytes
    streaming: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.status_code, bool)
            or not isinstance(self.status_code, int)
            or not 100 <= self.status_code <= 599
        ):
            raise ValueError("status_codeがHTTP statusの範囲外です")
        for name in ("content_type", "cache_control"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or "\r" in value or "\n" in value:
                raise ValueError(f"{name}が不正です")
        if not isinstance(self.body, bytes):
            raise TypeError("bodyはbytesである必要があります")
        if not isinstance(self.streaming, bool):
            raise TypeError("streamingはboolである必要があります")

    @property
    def headers(self) -> Mapping[str, str]:
        """live serverへそのまま渡せるimmutable headerを返す。"""
        headers = {
            "Content-Type": self.content_type,
            "Cache-Control": self.cache_control,
        }
        if not self.streaming:
            headers["Content-Length"] = str(len(self.body))
        return MappingProxyType(headers)


_ROUTES: Mapping[str, V1Route] = MappingProxyType({route.value: route for route in V1Route})


def require_loopback_bind(host: str) -> str:
    """hostnameや外部interfaceを拒否し、許可済みloopback literalを返す。"""
    if not isinstance(host, str):
        raise TypeError("bind hostは文字列である必要があります")
    if host not in ALLOWED_LOOPBACK_BINDS:
        raise ValueError("bind hostは127.0.0.1に限定されています")
    return host


def resolve_route(method: str, path: str) -> V1Route | None:
    """GETとqueryなしv1 pathの完全一致だけをrouteへ解決する。"""
    if not isinstance(method, str) or not isinstance(path, str):
        raise TypeError("methodとpathは文字列である必要があります")
    if method != "GET":
        return None
    return _ROUTES.get(path)


def _response(
    content_type: str,
    cache_control: str,
    body: bytes,
    *,
    streaming: bool = False,
) -> ResponseSpec:
    return ResponseSpec(HTTP_OK, content_type, cache_control, body, streaming)


def _require_valid_snapshot(snapshot: OverlaySnapshot) -> None:
    if not isinstance(snapshot, OverlaySnapshot):
        raise TypeError("snapshotはOverlaySnapshotである必要があります")
    report = validate_snapshot(snapshot)
    if not report.is_valid:
        raise PhaseJValidationError(report)


def _require_valid_health(health: OverlayHealth) -> None:
    if not isinstance(health, OverlayHealth):
        raise TypeError("healthはOverlayHealthである必要があります")
    report = validate_health(health)
    if not report.is_valid:
        raise PhaseJValidationError(report)


def build_overlay_response(html: str = V1_OVERLAY_HTML) -> ResponseSpec:
    """v1 EventSourceを含むoverlay HTML responseを構築する。"""
    if not isinstance(html, str):
        raise TypeError("htmlは文字列である必要があります")
    return _response(HTML_CONTENT_TYPE, NO_STORE, html.encode("utf-8"))


def build_events_response() -> ResponseSpec:
    """bodyを書き続けるSSE endpointの初期responseを構築する。"""
    return _response(SSE_CONTENT_TYPE, SSE_CACHE_CONTROL, b"", streaming=True)


def build_latest_response(snapshot: OverlaySnapshot) -> ResponseSpec:
    """検証済みlatest snapshotをcanonical JSON responseにする。"""
    _require_valid_snapshot(snapshot)
    body = serialize_snapshot(snapshot).encode("utf-8")
    return _response(JSON_CONTENT_TYPE, NO_STORE, body)


def build_health_response(health: OverlayHealth) -> ResponseSpec:
    """検証済みhealthをcanonical JSON responseにする。"""
    _require_valid_health(health)
    body = serialize_health(health).encode("utf-8")
    return _response(JSON_CONTENT_TYPE, NO_STORE, body)


def _event_id(snapshot: OverlaySnapshot) -> str:
    identity = snapshot.identity
    session_id = identity["session_id"]
    stream_seq = identity["stream_seq"]
    if "\r" in session_id or "\n" in session_id:
        raise ValueError("session_idにSSE改行を含められません")
    return f"{session_id}:{stream_seq}"


def encode_snapshot_event(snapshot: OverlaySnapshot) -> bytes:
    """snapshotをEventSource互換の単一SSE eventへencodeする。"""
    _require_valid_snapshot(snapshot)
    lines = (
        f"id: {_event_id(snapshot)}\n",
        "event: snapshot\n",
        f"data: {serialize_snapshot(snapshot)}\n",
        "\n",
    )
    return "".join(lines).encode("utf-8")


def encode_heartbeat() -> bytes:
    """event IDを消費しないSSE comment heartbeatを返す。"""
    return b": heartbeat\n\n"


__all__ = [
    "ALLOWED_LOOPBACK_BINDS",
    "DEFAULT_HEARTBEAT_INTERVAL_MS",
    "DEFAULT_SUBSCRIBER_LIMIT",
    "DEFAULT_WRITE_TIMEOUT_MS",
    "NO_STORE",
    "SSE_CACHE_CONTROL",
    "TransportPolicy",
    "ResponseSpec",
    "V1Route",
    "V1_OVERLAY_HTML",
    "build_events_response",
    "build_health_response",
    "build_latest_response",
    "build_overlay_response",
    "encode_heartbeat",
    "encode_snapshot_event",
    "require_loopback_bind",
    "resolve_route",
]
