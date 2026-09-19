"""確定攻撃と暫定攻撃を分離したcanonical観測正本V3。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from enum import StrEnum
from typing import Any, Mapping

from src.canonical_observation_v2 import (
    ACTIVE_COLORS,
    AvailabilityState,
    AvailableBool,
    AvailableColor,
    AvailableInt,
    AvailableSide,
    CanonicalObservationError,
    CanonicalObservationV2,
    CanonicalSideObservation,
    CausalLedgerSideSnapshot,
    CausalLedgerSnapshot,
    FieldProvenance,
    ImmutableGrid,
    ImmutableMask,
    ObservationProvenance,
    ObservationQuality,
    ObservationStatus,
    PiecePairObservation,
    PieceQueueObservation,
    ProvenanceKind,
    StableBoardObservation,
    validate_canonical_observation as validate_v2_observation,
)


CANONICAL_OBSERVATION_SCHEMA_VERSION = "canonical-observation/v3"
_V2_SCHEMA_VERSION = "canonical-observation/v2"
_AVAILABLE_STATES = frozenset({
    AvailabilityState.KNOWN, AvailabilityState.KNOWN_ZERO,
})
SIDES = ("p1", "p2")


class ProjectedDropProvenanceStatus(StrEnum):
    """projected receiptの検証可能性。"""

    VERIFIED = "verified"
    LEGACY_MISSING = "legacy_missing"


@dataclass(frozen=True, slots=True)
class ProjectedDropProvenanceV1:
    """guaranteed着弾内訳を導いたprojected DTOの不変receipt。"""

    status: ProjectedDropProvenanceStatus
    origin_recipient: str
    current_recipient: str
    input_digest: str | None
    adapter_hash: str | None
    physics_hash: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ProjectedDropProvenanceStatus):
            raise CanonicalObservationError("projected drop provenance statusが不正です")
        if self.origin_recipient not in SIDES or self.current_recipient not in SIDES:
            raise CanonicalObservationError("projected drop recipientが不正です")
        _validate_projected_hashes(self)

    @property
    def recipient(self) -> str:
        """旧呼出元向けのcurrent recipient alias。"""
        return self.current_recipient

    @classmethod
    def legacy_missing(cls, recipient: str) -> ProjectedDropProvenanceV1:
        """旧V3 payloadで保存されなかったreceiptを明示する。"""
        return cls(
            ProjectedDropProvenanceStatus.LEGACY_MISSING,
            recipient, recipient, None, None, None,
        )

    def swap_sides(self) -> ProjectedDropProvenanceV1:
        recipient = "p2" if self.current_recipient == "p1" else "p1"
        return replace(self, current_recipient=recipient)


@dataclass(frozen=True, slots=True)
class CanonicalObservationV3:
    """学習・shadow・本番が共用する、確定会計分離済み観測。"""

    observation_id: str
    source_video_id: str
    build_id: str
    game_idx: int
    through_event_seq: int
    cutoff_frame: int
    cutoff_ms: int
    boundary_segment: int
    p1: CanonicalSideObservation
    p2: CanonicalSideObservation
    ledger: CausalLedgerSnapshot
    provenance: ObservationProvenance
    quality: ObservationQuality
    projected_drop_provenance: ProjectedDropProvenanceV1 | None = None
    schema_version: str = CANONICAL_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CANONICAL_OBSERVATION_SCHEMA_VERSION:
            raise CanonicalObservationError("canonical observation V3 schema versionが不正です")
        validate_v2_observation(_as_v2(self))
        _validate_finalized_ledger(self.ledger)
        _validate_projected_drop_provenance(self)

    def to_dict(self) -> dict[str, Any]:
        """schema versionを含むJSON互換の完全表現を返す。"""
        return _json_mapping(asdict(self))

    def canonical_json_bytes(self) -> bytes:
        """キー順・空白・改行を固定した正規JSONを返す。"""
        return canonical_json_bytes(self)

    @property
    def digest(self) -> str:
        """canonical observation全体のSHA-256を返す。"""
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()

    @property
    def input_digest(self) -> str:
        """projected DTOと同名のdigest alias。"""
        return self.digest

    def swap_sides(self) -> CanonicalObservationV3:
        """全side slotを交換する。二回適用するとbit-identicalに戻る。"""
        projected = self.projected_drop_provenance
        return replace(
            self, p1=self.p2, p2=self.p1, ledger=self.ledger.swap_sides(),
            projected_drop_provenance=(
                None if projected is None else projected.swap_sides()
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CanonicalObservationV3:
        """未知keyを許さずJSON互換値から厳密に復元する。"""
        expected = {item.name for item in fields(cls)}
        legacy_expected = expected - {"projected_drop_provenance"}
        if not isinstance(value, Mapping) or set(value) not in (expected, legacy_expected):
            raise CanonicalObservationError("observationのkey集合がschemaと一致しません")
        if value.get("schema_version") != CANONICAL_OBSERVATION_SCHEMA_VERSION:
            raise CanonicalObservationError("canonical observation V3 schema versionが不正です")
        has_projected_receipt = "projected_drop_provenance" in value
        legacy = dict(value)
        raw_projected = legacy.pop("projected_drop_provenance", None)
        legacy["schema_version"] = _V2_SCHEMA_VERSION
        converted = CanonicalObservationV2.from_dict(legacy)
        projected = (
            _projected_drop_provenance_from_dict(raw_projected)
            if has_projected_receipt else _legacy_missing_projected_receipt(converted)
        )
        return _from_v2(converted, projected)

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> CanonicalObservationV3:
        """UTF-8 JSONから厳密に復元する。"""
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CanonicalObservationError("canonical observation JSONが不正です") from error
        if not isinstance(value, Mapping):
            raise CanonicalObservationError("canonical observationはJSON object必須です")
        return cls.from_dict(value)


def canonical_json_bytes(observation: CanonicalObservationV3) -> bytes:
    """CanonicalObservationV3だけを決定論的JSONへ変換する。"""
    validate_canonical_observation(observation)
    text = json.dumps(observation.to_dict(), ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":"))
    return (text + "\n").encode("utf-8")


def validate_canonical_observation(observation: CanonicalObservationV3) -> None:
    """公開境界で型偽装とV3確定会計不変条件を再検査する。"""
    if not isinstance(observation, CanonicalObservationV3):
        raise CanonicalObservationError("canonical observation V3の型が不正です")
    observation.__post_init__()


def _as_v2(value: CanonicalObservationV3) -> CanonicalObservationV2:
    return CanonicalObservationV2(
        observation_id=value.observation_id, source_video_id=value.source_video_id,
        build_id=value.build_id, game_idx=value.game_idx,
        through_event_seq=value.through_event_seq, cutoff_frame=value.cutoff_frame,
        cutoff_ms=value.cutoff_ms, boundary_segment=value.boundary_segment,
        p1=value.p1, p2=value.p2, ledger=value.ledger,
        provenance=value.provenance, quality=value.quality,
    )


def _from_v2(
    value: CanonicalObservationV2,
    projected: ProjectedDropProvenanceV1 | None = None,
) -> CanonicalObservationV3:
    return CanonicalObservationV3(
        observation_id=value.observation_id, source_video_id=value.source_video_id,
        build_id=value.build_id, game_idx=value.game_idx,
        through_event_seq=value.through_event_seq, cutoff_frame=value.cutoff_frame,
        cutoff_ms=value.cutoff_ms, boundary_segment=value.boundary_segment,
        p1=value.p1, p2=value.p2, ledger=value.ledger,
        provenance=value.provenance, quality=value.quality,
        projected_drop_provenance=projected,
    )


def _validate_finalized_ledger(ledger: CausalLedgerSnapshot) -> None:
    for side in (ledger.p1, ledger.p2):
        pending = side.pending_garbage
        if not (side.finalized_unsettled_attack == pending
                and side.post_cancel_residual == pending):
            raise CanonicalObservationError("V3確定pending/residualが同一prefix値ではありません")
        _validate_drop_semantics(side)


def _validate_drop_semantics(side: CausalLedgerSideSnapshot) -> None:
    pending = side.pending_garbage
    first = side.first_drop_amount
    leftover = side.leftover_after_first_drop
    if pending.availability not in _AVAILABLE_STATES:
        if first.availability != pending.availability or leftover.availability != pending.availability:
            raise CanonicalObservationError("非可用pendingと着弾内訳の状態が一致しません")
        return
    amount = int(pending.value)
    if amount == 0:
        if first != pending or leftover != pending:
            raise CanonicalObservationError("確定pending 0の着弾内訳はknown zero必須です")
        return
    _validate_positive_drop(amount, first, leftover)


def _validate_positive_drop(
    amount: int, first: AvailableInt, leftover: AvailableInt,
) -> None:
    states = (first.availability, leftover.availability)
    if states == (AvailabilityState.UNKNOWN, AvailabilityState.UNKNOWN):
        return
    if not all(state in _AVAILABLE_STATES for state in states):
        raise CanonicalObservationError("正の確定pendingの着弾内訳availabilityが不正です")
    if int(first.value) + int(leftover.value) != amount:
        raise CanonicalObservationError("確定pendingと着弾内訳の合計が一致しません")
    if any(item.provenance.kind != ProvenanceKind.RULE_DERIVED for item in (first, leftover)):
        raise CanonicalObservationError("既知の着弾内訳はguaranteed rule由来必須です")


def _validate_projected_drop_provenance(observation: CanonicalObservationV3) -> None:
    known_sides = tuple(
        side for side in SIDES
        if _has_known_positive_drop(getattr(observation.ledger, side))
    )
    projected = observation.projected_drop_provenance
    if not known_sides:
        if projected is not None:
            raise CanonicalObservationError("既知のguaranteed着弾なしにprovenanceがあります")
        return
    if len(known_sides) != 1 or projected is None:
        raise CanonicalObservationError("既知のguaranteed着弾にprojected provenanceがありません")
    if projected.current_recipient != known_sides[0]:
        raise CanonicalObservationError("projected provenanceのrecipientが不一致です")


def _has_known_positive_drop(side: CausalLedgerSideSnapshot) -> bool:
    if side.pending_garbage.availability not in _AVAILABLE_STATES:
        return False
    if int(side.pending_garbage.value) == 0:
        return False
    return all(
        item.availability in _AVAILABLE_STATES
        for item in (side.first_drop_amount, side.leftover_after_first_drop)
    )


def _projected_drop_provenance_from_dict(
    value: Any,
) -> ProjectedDropProvenanceV1 | None:
    if value is None:
        return None
    expected = {item.name for item in fields(ProjectedDropProvenanceV1)}
    legacy = {"recipient", "input_digest", "adapter_hash", "physics_hash"}
    if not isinstance(value, Mapping) or set(value) not in (expected, legacy):
        raise CanonicalObservationError("projected drop provenanceのkey集合が不正です")
    if set(value) == legacy:
        recipient = value["recipient"]
        return ProjectedDropProvenanceV1(
            ProjectedDropProvenanceStatus.VERIFIED, recipient, recipient,
            value["input_digest"], value["adapter_hash"], value["physics_hash"],
        )
    converted = dict(value)
    try:
        converted["status"] = ProjectedDropProvenanceStatus(converted["status"])
    except (TypeError, ValueError) as error:
        raise CanonicalObservationError("projected provenance statusが不正です") from error
    return ProjectedDropProvenanceV1(**converted)


def _legacy_missing_projected_receipt(
    observation: CanonicalObservationV2,
) -> ProjectedDropProvenanceV1 | None:
    known = tuple(
        side for side in SIDES
        if _has_known_positive_drop(getattr(observation.ledger, side))
    )
    return ProjectedDropProvenanceV1.legacy_missing(known[0]) if len(known) == 1 else None


def _validate_projected_hashes(value: ProjectedDropProvenanceV1) -> None:
    hashes = (value.input_digest, value.adapter_hash, value.physics_hash)
    if value.status == ProjectedDropProvenanceStatus.LEGACY_MISSING:
        if any(item is not None for item in hashes):
            raise CanonicalObservationError("legacy missing receiptにhashは保持できません")
        return
    for item, label in zip(hashes, (
        "projected input digest", "projected adapter hash", "projected physics hash",
    ), strict=True):
        _require_sha256(item, label)


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise CanonicalObservationError(f"{label}はSHA-256必須です")
    if any(character not in "0123456789abcdef" for character in value):
        raise CanonicalObservationError(f"{label}は小文字hex SHA-256必須です")


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _json_value(item) for key, item in value.items()}


def _json_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


__all__ = [
    "ACTIVE_COLORS", "AvailabilityState", "AvailableBool", "AvailableColor",
    "AvailableInt", "AvailableSide", "CANONICAL_OBSERVATION_SCHEMA_VERSION",
    "CanonicalObservationError", "CanonicalObservationV3", "CanonicalSideObservation",
    "CausalLedgerSideSnapshot", "CausalLedgerSnapshot", "FieldProvenance",
    "ImmutableGrid", "ImmutableMask", "ObservationProvenance", "ObservationQuality",
    "ObservationStatus", "PiecePairObservation", "PieceQueueObservation",
    "ProjectedDropProvenanceStatus", "ProjectedDropProvenanceV1", "ProvenanceKind",
    "StableBoardObservation", "canonical_json_bytes",
    "validate_canonical_observation",
]
