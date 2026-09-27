"""Phase J 公開snapshotのDTOと列挙契約。"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Self

SNAPSHOT_SCHEMA_VERSION = "puyo-overlay-snapshot/v1"
HEALTH_SCHEMA_VERSION = "puyo-overlay-health/v1"
ENUM_MANIFEST_VERSION = "puyo-overlay-enums/v1"

PREDICTION_DISCARD_REASONS = frozenset({
    'unknown_cells', 'observed_exceeds_prediction', 'confirmed_board_mismatch'})


@dataclass(frozen=True)
class DisplayLayers:
    """C案の追加契約。現在値は最後の確定盤面評価、予測値は未取得ならnull。"""

    current_p1: float | None
    predicted_p1: float | None
    displayed_p1: float | None
    includes_prediction: bool
    prediction_discard_reason: str | None = None

    def __post_init__(self) -> None:
        for value in (self.current_p1, self.predicted_p1, self.displayed_p1):
            if value is not None and (isinstance(value, bool) or
                    not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('層別勝率は有限の0〜1またはnullが必要です')
        if type(self.includes_prediction) is not bool:
            raise ValueError('予測込みの印はboolが必要です')
        if self.includes_prediction and (self.predicted_p1 is None or self.displayed_p1 is None):
            raise ValueError('予測込み表示には予測値と表示値が必要です')
        if self.prediction_discard_reason not in PREDICTION_DISCARD_REASONS | {None}:
            raise ValueError('未知の予測破棄理由です')
        if self.prediction_discard_reason and (self.includes_prediction or self.current_p1 is None):
            raise ValueError('予測破棄後は現在層へ戻る必要があります')


class Visibility(StrEnum):
    VISIBLE = "visible"
    HIDDEN = "hidden"


class DisplayStatus(StrEnum):
    LIVE = "live"
    PHYSICAL_PREDICTION = "physical_prediction"
    HOLD = "hold"
    TERMINAL_FACT = "terminal_fact"
    RESULT = "result"
    WAITING = "waiting"
    INTEGRITY_FAULT = "integrity_fault"


class UpdateReason(StrEnum):
    INITIAL_SNAPSHOT = "initial_snapshot"
    OBSERVATION_UPDATE = "observation_update"
    PHYSICAL_FACT_UPDATE = "physical_fact_update"
    PREDICTION_COMMITTED = "prediction_committed"
    HOLD_STARTED = "hold_started"
    HOLD_REASON_CHANGED = "hold_reason_changed"
    HOLD_EXPIRED = "hold_expired"
    TERMINAL_CONFIRMED = "terminal_confirmed"
    RESULT_TRANSITION = "result_transition"
    FORMAL_BOUNDARY = "formal_boundary"
    CONFIG_PENDING = "config_pending"
    WORKER_STATE_CHANGED = "worker_state_changed"
    TIMER_ELAPSED = "timer_elapsed"
    INTEGRITY_FAULT = "integrity_fault"


class HoldReason(StrEnum):
    TERMINAL_CONFIRMATION_PENDING = "terminal_confirmation_pending"
    RECOGNITION_UNRELIABLE = "recognition_unreliable"
    PHYSICS_AMBIGUOUS = "physics_ambiguous"
    PREDICTION_WORKER_FAULT = "prediction_worker_fault"
    PREDICTION_DEADLINE_MISSED = "prediction_deadline_missed"
    CALCULATION_PENDING = "calculation_pending"
    BOTH_CHAINING_NO_NEW_FACT = "both_chaining_no_new_fact"


class IntegrityStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    FAULT = "fault"


class FaultCode(StrEnum):
    SEQUENCE_GAP = "sequence_gap"
    SEQUENCE_REVERSED = "sequence_reversed"
    ID_CONTENT_CONFLICT = "id_content_conflict"
    ASSET_BUNDLE_CHANGED_MID_MATCH = "asset_bundle_changed_mid_match"
    SCHEMA_INVALID = "schema_invalid"
    CUTOFF_VIOLATION = "cutoff_violation"
    SNAPSHOT_REVISION_STALE = "snapshot_revision_stale"
    UNEXPECTED_BOUNDARY = "unexpected_boundary"
    INTERNAL_INVARIANT_FAILED = "internal_invariant_failed"


class EvaluationMode(StrEnum):
    PRACTICAL = "practical"
    BEST_ACTION = "best_action"
    BOTH = "both"


class EvaluationAvailability(StrEnum):
    AVAILABLE = "available"
    PENDING = "pending"
    UNAVAILABLE = "unavailable"


class PracticalOrigin(StrEnum):
    OBSERVED = "observed"
    PHYSICAL_PREDICTION = "physical_prediction"
    MODEL = "model"


class BoardProvenance(StrEnum):
    CONFIRMED = "confirmed"
    PHYSICS_PROJECTED = "physics_projected"
    UNKNOWN = "unknown"


class RecognitionQualityStatus(StrEnum):
    TRUSTED = "trusted"
    PARTIAL = "partial"
    UNTRUSTED = "untrusted"


class RecognitionReason(StrEnum):
    BOARD_UNSTABLE = "board_unstable"
    BOARD_CONFLICT = "board_conflict"
    NEXT_UNKNOWN = "next_unknown"
    CAPTURE_GAP = "capture_gap"
    CALIBRATION_UNAVAILABLE = "calibration_unavailable"
    RECOGNITION_ASSET_MISMATCH = "recognition_asset_mismatch"


class UnresolvedPhysicsReason(StrEnum):
    CHAIN_RESOLVING = "chain_resolving"
    CANCEL_UNRESOLVED = "cancel_unresolved"
    GARBAGE_PENDING = "garbage_pending"
    LANDING_AMBIGUOUS = "landing_ambiguous"
    TERMINAL_CANDIDATE_PENDING = "terminal_candidate_pending"


class TerminalState(StrEnum):
    NONE = "none"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"


class TerminalWinner(StrEnum):
    PLAYER_1 = "1P"
    PLAYER_2 = "2P"


class TerminalEvidenceKind(StrEnum):
    VISUAL_RESULT_LOGO_BILATERAL_2X2 = "visual_result_logo_bilateral_2x2"


class TerminalResultCode(StrEnum):
    PLAYER_1_WIN = "p1_win"
    PLAYER_2_WIN = "p2_win"


class RuntimeStatus(StrEnum):
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"


class RuntimeTier(StrEnum):
    LIGHTWEIGHT = "lightweight"
    STANDARD = "standard"
    HIGH_ACCURACY = "high_accuracy"
    ANALYSIS = "analysis"


class WorkerHealth(StrEnum):
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RESTARTING = "restarting"
    DISABLED = "disabled"


class TelemetryHealth(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class HealthErrorCode(StrEnum):
    TELEMETRY_SINK_DEGRADED = "telemetry_sink_degraded"
    TELEMETRY_SINK_UNAVAILABLE = "telemetry_sink_unavailable"
    PRACTICAL_WORKER_FAULT = "practical_worker_fault"
    BEST_ACTION_WORKER_FAULT = "best_action_worker_fault"
    SNAPSHOT_STALE = "snapshot_stale"
    INTERNAL_HEALTH_INVARIANT_FAILED = "internal_health_invariant_failed"


ENUM_TYPES: Mapping[str, type[StrEnum]] = MappingProxyType(
    {
        "visibility": Visibility,
        "display_status": DisplayStatus,
        "update_reason": UpdateReason,
        "hold_reason": HoldReason,
        "integrity_status": IntegrityStatus,
        "fault_code": FaultCode,
        "evaluation_mode": EvaluationMode,
        "evaluation_availability": EvaluationAvailability,
        "practical_origin": PracticalOrigin,
        "board_provenance": BoardProvenance,
        "recognition_quality_status": RecognitionQualityStatus,
        "recognition_reason": RecognitionReason,
        "unresolved_physics_reason": UnresolvedPhysicsReason,
        "terminal_state": TerminalState,
        "terminal_winner": TerminalWinner,
        "terminal_evidence_kind": TerminalEvidenceKind,
        "terminal_result_code": TerminalResultCode,
        "runtime_status": RuntimeStatus,
        "runtime_tier": RuntimeTier,
        "worker_health": WorkerHealth,
        "telemetry_health": TelemetryHealth,
        "health_error_code": HealthErrorCode,
    }
)

NULLABLE_ENUM_NAMES = frozenset(
    {
        "hold_reason",
        "practical_origin",
        "terminal_winner",
        "terminal_evidence_kind",
        "terminal_result_code",
        "health_error_code",
    }
)


def _freeze_json(value: Any) -> Any:
    """JSON互換値を再帰的に読取専用へ変換する。"""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    copied = deepcopy(dict(value))
    frozen = _freeze_json(copied)
    if not isinstance(frozen, Mapping):  # pragma: no cover - 内部不変条件
        raise TypeError("mappingの凍結結果がmappingではありません")
    return frozen


def _thaw_json(value: Any) -> Any:
    """読取専用値を独立したJSON互換値へ戻す。"""
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return deepcopy(value)


def _serialize_mapping(payload: Mapping[str, Any]) -> str:
    """mappingをキー順に依存しない決定的JSONへ直列化する。"""
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _mapping_content_digest(payload: Mapping[str, Any]) -> str:
    serialized = _serialize_mapping(payload)
    digest = sha256(serialized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True, slots=True)
class OverlaySnapshot:
    """Phase J公開snapshot。値域と相関制約の検査はvalidatorへ委譲する。"""

    schema_version: str
    identity: Mapping[str, Any]
    timing: Mapping[str, Any]
    display: Mapping[str, Any]
    integrity: Mapping[str, Any]
    mode: Mapping[str, Any]
    evaluations: Mapping[str, Any]
    input: Mapping[str, Any]
    terminal: Mapping[str, Any]
    runtime: Mapping[str, Any]
    assets: Mapping[str, Any]
    unknown_fields: Mapping[str, Any] = field(default_factory=dict, repr=False)

    ROOT_FIELDS: ClassVar[tuple[str, ...]] = (
        "schema_version",
        "identity",
        "timing",
        "display",
        "integrity",
        "mode",
        "evaluations",
        "input",
        "terminal",
        "runtime",
        "assets",
    )
    MAPPING_FIELDS: ClassVar[tuple[str, ...]] = ROOT_FIELDS[1:]

    def __post_init__(self) -> None:
        overlap = set(self.unknown_fields).intersection(self.ROOT_FIELDS)
        if overlap:
            joined = ", ".join(sorted(overlap))
            raise ValueError(f"unknown_fieldsが既知fieldと重複しています: {joined}")
        for name in (*self.MAPPING_FIELDS, "unknown_fields"):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name)))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Self:
        """mappingから必須root fieldと未知fieldを損失なく分離する。"""
        copied = deepcopy(dict(payload))
        missing = [name for name in cls.ROOT_FIELDS if name not in copied]
        if missing:
            raise ValueError(f"必須root fieldがありません: {', '.join(missing)}")
        known = {name: copied.pop(name) for name in cls.ROOT_FIELDS}
        return cls(**known, unknown_fields=copied)

    @classmethod
    def deserialize(cls, serialized: str | bytes | bytearray) -> Self:
        """JSON文字列からsnapshot DTOを復元する。"""
        payload = json.loads(serialized)
        if not isinstance(payload, Mapping):
            raise ValueError("snapshotのrootはobjectである必要があります")
        return cls.from_mapping(payload)

    def to_mapping(self) -> dict[str, Any]:
        """未知fieldを含む独立したJSON互換mappingを返す。"""
        payload = _thaw_json(self.unknown_fields)
        payload.update(
            {
                "schema_version": self.schema_version,
                **{name: _thaw_json(getattr(self, name)) for name in self.MAPPING_FIELDS},
            }
        )
        return payload

    def serialize(self) -> str:
        """キー順に依存しない決定的JSONへ直列化する。"""
        return _serialize_mapping(self.to_mapping())

    def content_digest(self) -> str:
        """決定的JSONのSHA-256 digestを返す。"""
        return _mapping_content_digest(self.to_mapping())


@dataclass(frozen=True, slots=True)
class OverlayHealth:
    """Phase J health endpointの公開DTO。"""

    schema_version: str
    server_status: str
    session_id: str
    latest_stream_seq: int
    latest_snapshot_age_ms: int | float
    subscriber_count: int
    subscriber_limit: int
    telemetry_health: str
    bound_host: str
    bound_port: int
    started_at_utc: str
    checked_at_utc: str
    last_error_code: str | None
    unknown_fields: Mapping[str, Any] = field(default_factory=dict, repr=False)

    ROOT_FIELDS: ClassVar[tuple[str, ...]] = (
        "schema_version",
        "server_status",
        "session_id",
        "latest_stream_seq",
        "latest_snapshot_age_ms",
        "subscriber_count",
        "subscriber_limit",
        "telemetry_health",
        "bound_host",
        "bound_port",
        "started_at_utc",
        "checked_at_utc",
        "last_error_code",
    )

    def __post_init__(self) -> None:
        overlap = set(self.unknown_fields).intersection(self.ROOT_FIELDS)
        if overlap:
            joined = ", ".join(sorted(overlap))
            raise ValueError(f"unknown_fieldsが既知fieldと重複しています: {joined}")
        object.__setattr__(self, "unknown_fields", _freeze_mapping(self.unknown_fields))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Self:
        """mappingから必須root fieldと未知fieldを損失なく分離する。"""
        copied = deepcopy(dict(payload))
        missing = [name for name in cls.ROOT_FIELDS if name not in copied]
        if missing:
            raise ValueError(f"必須root fieldがありません: {', '.join(missing)}")
        known = {name: copied.pop(name) for name in cls.ROOT_FIELDS}
        return cls(**known, unknown_fields=copied)

    @classmethod
    def deserialize(cls, serialized: str | bytes | bytearray) -> Self:
        """JSON文字列からhealth DTOを復元する。"""
        payload = json.loads(serialized)
        if not isinstance(payload, Mapping):
            raise ValueError("healthのrootはobjectである必要があります")
        return cls.from_mapping(payload)

    def to_mapping(self) -> dict[str, Any]:
        """未知fieldを含む独立したJSON互換mappingを返す。"""
        payload = _thaw_json(self.unknown_fields)
        payload.update({name: getattr(self, name) for name in self.ROOT_FIELDS})
        return payload

    def serialize(self) -> str:
        """キー順に依存しない決定的JSONへ直列化する。"""
        return _serialize_mapping(self.to_mapping())

    def content_digest(self) -> str:
        """決定的JSONのSHA-256 digestを返す。"""
        return _mapping_content_digest(self.to_mapping())


def serialize_snapshot(snapshot: OverlaySnapshot) -> str:
    """公開helperとしてsnapshotを決定的JSONへ直列化する。"""
    return snapshot.serialize()


def deserialize_snapshot(serialized: str | bytes | bytearray) -> OverlaySnapshot:
    """公開helperとしてJSONからsnapshotを復元する。"""
    return OverlaySnapshot.deserialize(serialized)


def snapshot_content_digest(snapshot: OverlaySnapshot) -> str:
    """公開helperとしてsnapshotのcontent digestを返す。"""
    return snapshot.content_digest()


def serialize_health(health: OverlayHealth) -> str:
    """公開helperとしてhealthを決定的JSONへ直列化する。"""
    return health.serialize()


def deserialize_health(serialized: str | bytes | bytearray) -> OverlayHealth:
    """公開helperとしてJSONからhealthを復元する。"""
    return OverlayHealth.deserialize(serialized)


def health_content_digest(health: OverlayHealth) -> str:
    """公開helperとしてhealthのcontent digestを返す。"""
    return health.content_digest()
