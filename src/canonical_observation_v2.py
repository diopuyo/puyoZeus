"""次期有利不利モデルが学習・shadow・本番で共用する観測正本V2。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields, replace
from enum import StrEnum
from typing import Any, Mapping, TypeVar

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_UNKNOWN, VALID_COLORS
from src.scoring import OJAMA_MAX_DROP_PER_TURN


CANONICAL_OBSERVATION_SCHEMA_VERSION = "canonical-observation/v2"
SIDES = frozenset({"p1", "p2"})
SIDE_VALUES = frozenset({"p1", "p2", "both"})
ACTIVE_COLORS = frozenset({1, 2, 3, 4, 5})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

ImmutableGrid = tuple[tuple[int, ...], ...]
ImmutableMask = tuple[tuple[bool, ...], ...]


class CanonicalObservationError(ValueError):
    """CanonicalObservationV2の型・因果・完全性契約に違反した。"""


class AvailabilityState(StrEnum):
    """値の0と非可用理由を混同しない固定状態。"""

    KNOWN = "known"
    KNOWN_ZERO = "known_zero"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    INTEGRITY_FAULT = "integrity_fault"


class ProvenanceKind(StrEnum):
    """値を得た方法または非可用の原因区分。"""

    DIRECT_OBSERVATION = "direct_observation"
    CAUSAL_LEDGER = "causal_ledger"
    RULE_DERIVED = "rule_derived"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    INTEGRITY_FAULT = "integrity_fault"


class ObservationStatus(StrEnum):
    """モデル入力前に適用するhard status。"""

    READY = "ready"
    HOLD = "hold"
    UNSUPPORTED = "unsupported"
    INTEGRITY_FAULT = "integrity_fault"


@dataclass(frozen=True, slots=True)
class FieldProvenance:
    """一つのraw値が予測時点までに得られた根拠。"""

    kind: ProvenanceKind
    source_event_ids: tuple[str, ...]
    available_frame: int | None
    available_ms: int | None
    confidence_milli: int | None = None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ProvenanceKind):
            raise CanonicalObservationError("provenance kindが固定Enumではありません")
        _require_text_tuple(self.source_event_ids, "source event IDs", allow_empty=True)
        _require_sorted_unique(self.source_event_ids, "source event IDs")
        _validate_optional_position(self.available_frame, self.available_ms)
        if (self.confidence_milli is not None
                and not _is_int_between(self.confidence_milli, 0, 1000)):
            raise CanonicalObservationError("confidence_milliは0..1000の整数です")
        _require_text_tuple(self.reason_codes, "reason codes", allow_empty=True)
        _require_sorted_unique(self.reason_codes, "reason codes")
        _validate_provenance_shape(self)

    @classmethod
    def known(
        cls, event_ids: tuple[str, ...], available_frame: int, available_ms: int,
        *, kind: ProvenanceKind = ProvenanceKind.DIRECT_OBSERVATION,
        confidence_milli: int | None = None,
    ) -> FieldProvenance:
        """観測・会計・物理導出で確定した値の由来を作る。"""
        return cls(kind, tuple(sorted(event_ids)), available_frame, available_ms,
                   confidence_milli, ())

    @classmethod
    def unavailable(
        cls, state: AvailabilityState, reason_codes: tuple[str, ...],
    ) -> FieldProvenance:
        """unknown/unsupported/integrity faultの由来を作る。"""
        kind = ProvenanceKind(state.value)
        return cls(kind, (), None, None, None, tuple(sorted(reason_codes)))


@dataclass(frozen=True, slots=True)
class AvailableInt:
    """非負整数とavailabilityを分離した値。"""

    value: int | None
    availability: AvailabilityState
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        _validate_available(self.value, self.availability, self.provenance, "整数値")
        if self.value is not None and (type(self.value) is not int or self.value < 0):
            raise CanonicalObservationError("available intは非負整数です")
        _validate_zero_state(self.value, self.availability, 0, "整数値")

    @classmethod
    def known(cls, value: int, provenance: FieldProvenance) -> AvailableInt:
        state = AvailabilityState.KNOWN_ZERO if value == 0 else AvailabilityState.KNOWN
        return cls(value, state, provenance)

    @classmethod
    def unavailable(
        cls, state: AvailabilityState, provenance: FieldProvenance,
    ) -> AvailableInt:
        return cls(None, state, provenance)


@dataclass(frozen=True, slots=True)
class AvailableBool:
    """Falseをknown_zeroとして保持するboolean値。"""

    value: bool | None
    availability: AvailabilityState
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        _validate_available(self.value, self.availability, self.provenance, "bool値")
        if self.value is not None and type(self.value) is not bool:
            raise CanonicalObservationError("available boolの値はboolです")
        _validate_zero_state(self.value, self.availability, False, "bool値")

    @classmethod
    def known(cls, value: bool, provenance: FieldProvenance) -> AvailableBool:
        state = AvailabilityState.KNOWN if value else AvailabilityState.KNOWN_ZERO
        return cls(value, state, provenance)

    @classmethod
    def unavailable(
        cls, state: AvailabilityState, provenance: FieldProvenance,
    ) -> AvailableBool:
        return cls(None, state, provenance)


@dataclass(frozen=True, slots=True)
class AvailableColor:
    """組ぷよ色1..5とavailabilityを分離したcategorical値。"""

    value: int | None
    availability: AvailabilityState
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        _validate_available(self.value, self.availability, self.provenance, "組ぷよ色")
        if self.availability == AvailabilityState.KNOWN_ZERO:
            raise CanonicalObservationError("組ぷよ色にknown_zeroは使えません")
        if self.value is not None and (type(self.value) is not int or self.value not in ACTIVE_COLORS):
            raise CanonicalObservationError("組ぷよ色は1..5です")

    @classmethod
    def known(cls, value: int, provenance: FieldProvenance) -> AvailableColor:
        return cls(value, AvailabilityState.KNOWN, provenance)

    @classmethod
    def unavailable(
        cls, state: AvailabilityState, provenance: FieldProvenance,
    ) -> AvailableColor:
        return cls(None, state, provenance)


@dataclass(frozen=True, slots=True)
class AvailableSide:
    """p1/p2/bothまたは「該当sideなし」を保持する値。"""

    value: str | None
    availability: AvailabilityState
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        _validate_available_side(self.value, self.availability, self.provenance)

    @classmethod
    def known(cls, value: str, provenance: FieldProvenance) -> AvailableSide:
        return cls(value, AvailabilityState.KNOWN, provenance)

    @classmethod
    def known_none(cls, provenance: FieldProvenance) -> AvailableSide:
        return cls(None, AvailabilityState.KNOWN_ZERO, provenance)

    @classmethod
    def unavailable(
        cls, state: AvailabilityState, provenance: FieldProvenance,
    ) -> AvailableSide:
        return cls(None, state, provenance)

    def swap_sides(self) -> AvailableSide:
        if self.value is None or self.value == "both":
            return self
        return replace(self, value=_opponent(self.value))


@dataclass(frozen=True, slots=True)
class StableBoardObservation:
    """最新STABLEのraw 13x6盤面とcell known mask。"""

    grid: ImmutableGrid | None
    known_mask: ImmutableMask | None
    availability: AvailabilityState
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        _validate_available(self.grid, self.availability, self.provenance, "STABLE盤面")
        if self.availability == AvailabilityState.KNOWN_ZERO:
            raise CanonicalObservationError("盤面にknown_zeroは使えません")
        if self.availability == AvailabilityState.KNOWN:
            _validate_grid(self.grid)
            _validate_known_mask(self.grid, self.known_mask)
        elif self.grid is not None or self.known_mask is not None:
            raise CanonicalObservationError("非可用盤面はgrid/mask=None必須です")

    @classmethod
    def known(
        cls, grid: ImmutableGrid, known_mask: ImmutableMask,
        provenance: FieldProvenance,
    ) -> StableBoardObservation:
        return cls(grid, known_mask, AvailabilityState.KNOWN, provenance)

    @classmethod
    def unavailable(
        cls, state: AvailabilityState, provenance: FieldProvenance,
    ) -> StableBoardObservation:
        return cls(None, None, state, provenance)

    @property
    def source_event_id(self) -> str | None:
        ids = self.provenance.source_event_ids
        return ids[0] if len(ids) == 1 else None


@dataclass(frozen=True, slots=True)
class PiecePairObservation:
    """軸・子を個別availability付きで保持するraw組ぷよ。"""

    axis: AvailableColor
    child: AvailableColor

    def __post_init__(self) -> None:
        if not isinstance(self.axis, AvailableColor) or not isinstance(self.child, AvailableColor):
            raise CanonicalObservationError("組ぷよpairの型が不正です")


@dataclass(frozen=True, slots=True)
class PieceQueueObservation:
    """current/NEXT/DNEXTを派生フラグへ潰さず保持する。"""

    current: PiecePairObservation
    next: PiecePairObservation
    double_next: PiecePairObservation

    def __post_init__(self) -> None:
        values = (self.current, self.next, self.double_next)
        if any(not isinstance(value, PiecePairObservation) for value in values):
            raise CanonicalObservationError("piece queueの型が不正です")


@dataclass(frozen=True, slots=True)
class CanonicalSideObservation:
    """一側のraw current stateと監査専用文脈。"""

    board: StableBoardObservation
    pieces: PieceQueueObservation
    all_clear_pending: AvailableBool
    raw_score: AvailableInt
    tsumo_count: AvailableInt

    def __post_init__(self) -> None:
        expected = (StableBoardObservation, PieceQueueObservation, AvailableBool,
                    AvailableInt, AvailableInt)
        actual = (self.board, self.pieces, self.all_clear_pending,
                  self.raw_score, self.tsumo_count)
        if any(not isinstance(value, kind) for value, kind in zip(actual, expected, strict=True)):
            raise CanonicalObservationError("side observationの型が不正です")


@dataclass(frozen=True, slots=True)
class CausalLedgerSideSnapshot:
    """一側に属する未解決物理量。値は全てrawで保持する。"""

    pending_garbage: AvailableInt
    effective_rate: AvailableInt
    chain_active: AvailableBool
    chain_step: AvailableInt
    provisional_generated: AvailableInt
    provisional_score: AvailableInt
    provisional_chain_count: AvailableInt
    finalized_unsettled_attack: AvailableInt
    post_cancel_residual: AvailableInt
    send_waiting: AvailableBool
    placement_waiting: AvailableBool
    chain_end_waiting: AvailableBool
    garbage_falling: AvailableBool
    first_drop_amount: AvailableInt
    leftover_after_first_drop: AvailableInt

    def __post_init__(self) -> None:
        _validate_ledger_side_types(self)
        if self.effective_rate.availability == AvailabilityState.KNOWN_ZERO:
            raise CanonicalObservationError("known effective rateは正整数必須です")
        _validate_first_drop(self)


@dataclass(frozen=True, slots=True)
class CausalLedgerSnapshot:
    """同一event prefixから復元した左右対称の因果会計snapshot。"""

    p1: CausalLedgerSideSnapshot
    p2: CausalLedgerSideSnapshot
    active_chain_side: AvailableSide
    recipient: AvailableSide
    ledger_prefix_digest: str
    provenance: FieldProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.p1, CausalLedgerSideSnapshot) or not isinstance(
            self.p2, CausalLedgerSideSnapshot,
        ):
            raise CanonicalObservationError("ledger side snapshotの型が不正です")
        if not isinstance(self.active_chain_side, AvailableSide) or not isinstance(
            self.recipient, AvailableSide,
        ):
            raise CanonicalObservationError("ledger side値の型が不正です")
        _require_sha256(self.ledger_prefix_digest, "ledger prefix digest")
        if not isinstance(self.provenance, FieldProvenance):
            raise CanonicalObservationError("ledger provenanceの型が不正です")
        _validate_known_provenance(self.provenance, "ledger snapshot")
        _validate_active_chain_side(self)
        _validate_recipient(self)

    def swap_sides(self) -> CausalLedgerSnapshot:
        return replace(
            self, p1=self.p2, p2=self.p1,
            active_chain_side=self.active_chain_side.swap_sides(),
            recipient=self.recipient.swap_sides(),
        )


@dataclass(frozen=True, slots=True)
class ObservationProvenance:
    """canonical observation全体のlineage。"""

    event_prefix_semantic_digest: str
    source_group_id: str | None = None
    attempt_id: str | None = None
    recognition_code_sha256: str | None = None
    adapter_code_sha256: str | None = None
    manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.event_prefix_semantic_digest, "event prefix semantic digest")
        for value, label in ((self.source_group_id, "source group ID"),
                             (self.attempt_id, "attempt ID")):
            if value is not None:
                _require_text(value, label)
        for value, label in ((self.recognition_code_sha256, "recognition code SHA"),
                             (self.adapter_code_sha256, "adapter code SHA"),
                             (self.manifest_sha256, "manifest SHA")):
            if value is not None:
                _require_sha256(value, label)


@dataclass(frozen=True, slots=True)
class ObservationQuality:
    """値branchへ混ぜないhard statusと監査理由。"""

    status: ObservationStatus
    reason_codes: tuple[str, ...]
    quarantined: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, ObservationStatus):
            raise CanonicalObservationError("quality statusが固定Enumではありません")
        _require_text_tuple(self.reason_codes, "quality reason codes", allow_empty=True)
        _require_sorted_unique(self.reason_codes, "quality reason codes")
        if type(self.quarantined) is not bool:
            raise CanonicalObservationError("quality quarantinedはbool必須です")
        if self.status != ObservationStatus.READY and not self.reason_codes:
            raise CanonicalObservationError("非READY qualityにはreason codeが必要です")
        if self.status == ObservationStatus.READY and self.quarantined:
            raise CanonicalObservationError("READY observationはquarantineできません")
        if self.status == ObservationStatus.INTEGRITY_FAULT and not self.quarantined:
            raise CanonicalObservationError("integrity faultはquarantine必須です")


@dataclass(frozen=True, slots=True)
class CanonicalObservationV2:
    """学習・shadow・本番推論で共用する唯一の観測正本。"""

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
    schema_version: str = CANONICAL_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_observation_identity(self)
        _validate_observation_types(self)
        _validate_causal_cutoff(self)
        _validate_observation_quality(self)

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
        """既存projected DTOと同名のdigest alias。"""
        return self.digest

    def swap_sides(self) -> CanonicalObservationV2:
        """全side slotを交換する。二回適用するとbit-identicalに戻る。"""
        return replace(self, p1=self.p2, p2=self.p1, ledger=self.ledger.swap_sides())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CanonicalObservationV2:
        """未知keyを許さずJSON互換値から厳密に復元する。"""
        return _observation_from_dict(value)

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> CanonicalObservationV2:
        """UTF-8 JSONから厳密に復元する。"""
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CanonicalObservationError("canonical observation JSONが不正です") from error
        if not isinstance(value, Mapping):
            raise CanonicalObservationError("canonical observationはJSON object必須です")
        return cls.from_dict(value)


def canonical_json_bytes(observation: CanonicalObservationV2) -> bytes:
    """CanonicalObservationV2だけを決定論的JSONへ変換する。"""
    validate_canonical_observation(observation)
    text = json.dumps(observation.to_dict(), ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":"))
    return (text + "\n").encode("utf-8")


def validate_canonical_observation(observation: CanonicalObservationV2) -> None:
    """公開境界で型偽装を拒否しdataclass不変条件を再検査する。"""
    if not isinstance(observation, CanonicalObservationV2):
        raise CanonicalObservationError("canonical observationの型が不正です")
    observation.__post_init__()
    for nested in _nested_validatables(observation):
        nested.__post_init__()


def _validate_available(
    value: object, state: AvailabilityState, provenance: FieldProvenance, label: str,
) -> None:
    if not isinstance(state, AvailabilityState) or not isinstance(provenance, FieldProvenance):
        raise CanonicalObservationError(f"{label}のavailability/provenance型が不正です")
    if state in {AvailabilityState.KNOWN, AvailabilityState.KNOWN_ZERO}:
        if value is None:
            raise CanonicalObservationError(f"known {label}に値がありません")
        if provenance.kind not in _KNOWN_PROVENANCE_KINDS:
            raise CanonicalObservationError(f"known {label}のprovenanceが不正です")
        if not provenance.source_event_ids or provenance.available_frame is None:
            raise CanonicalObservationError(f"known {label}のevent/timingがありません")
        return
    if value is not None:
        raise CanonicalObservationError(f"非可用{label}はvalue=None必須です")
    if provenance.kind.value != state.value or not provenance.reason_codes:
        raise CanonicalObservationError(f"非可用{label}の理由と状態が一致しません")


_KNOWN_PROVENANCE_KINDS = frozenset({
    ProvenanceKind.DIRECT_OBSERVATION,
    ProvenanceKind.CAUSAL_LEDGER,
    ProvenanceKind.RULE_DERIVED,
})


def _validate_zero_state(
    value: object, state: AvailabilityState, zero: object, label: str,
) -> None:
    if value == zero and state != AvailabilityState.KNOWN_ZERO:
        raise CanonicalObservationError(f"既知0の{label}はknown_zero必須です")
    if value is not None and value != zero and state != AvailabilityState.KNOWN:
        raise CanonicalObservationError(f"非0の{label}はknown必須です")


def _validate_available_side(
    value: str | None, state: AvailabilityState, provenance: FieldProvenance,
) -> None:
    if not isinstance(state, AvailabilityState) or not isinstance(provenance, FieldProvenance):
        raise CanonicalObservationError("side値のavailability/provenance型が不正です")
    if state == AvailabilityState.KNOWN_ZERO:
        if value is not None:
            raise CanonicalObservationError("known_zero sideはvalue=None必須です")
        _validate_known_provenance(provenance, "known_zero side")
        return
    _validate_available(value, state, provenance, "side値")
    if value is not None and value not in SIDE_VALUES:
        raise CanonicalObservationError("side値はp1/p2/bothです")


def _validate_known_provenance(provenance: FieldProvenance, label: str) -> None:
    if not isinstance(provenance, FieldProvenance) or provenance.kind not in _KNOWN_PROVENANCE_KINDS:
        raise CanonicalObservationError(f"{label}のprovenanceが不正です")
    if not provenance.source_event_ids or provenance.available_frame is None:
        raise CanonicalObservationError(f"{label}のevent/timingがありません")


def _validate_grid(grid: ImmutableGrid | None) -> None:
    if type(grid) is not tuple or len(grid) != BOARD_ROWS:
        raise CanonicalObservationError("盤面はtupleの13行必須です")
    for row in grid:
        if type(row) is not tuple or len(row) != BOARD_COLS:
            raise CanonicalObservationError("盤面は各行tupleの6列必須です")
        if any(type(cell) is not int or cell not in VALID_COLORS for cell in row):
            raise CanonicalObservationError("盤面cellが正式色値ではありません")


def _validate_known_mask(grid: ImmutableGrid | None, mask: ImmutableMask | None) -> None:
    if type(mask) is not tuple or len(mask) != BOARD_ROWS:
        raise CanonicalObservationError("known maskはtupleの13行必須です")
    for grid_row, mask_row in zip(grid, mask, strict=True):
        if type(mask_row) is not tuple or len(mask_row) != BOARD_COLS:
            raise CanonicalObservationError("known maskは各行tupleの6列必須です")
        if any(type(cell) is not bool for cell in mask_row):
            raise CanonicalObservationError("known mask cellはbool必須です")
        if any(flag != (cell != COLOR_UNKNOWN) for cell, flag in zip(grid_row, mask_row)):
            raise CanonicalObservationError("gridとknown maskが一致しません")


def _validate_ledger_side_types(value: CausalLedgerSideSnapshot) -> None:
    bool_names = {"chain_active", "send_waiting", "placement_waiting",
                  "chain_end_waiting", "garbage_falling"}
    for item in fields(value):
        expected = AvailableBool if item.name in bool_names else AvailableInt
        if not isinstance(getattr(value, item.name), expected):
            raise CanonicalObservationError(f"ledger {item.name}の型が不正です")


def _validate_first_drop(value: CausalLedgerSideSnapshot) -> None:
    residual = value.post_cancel_residual
    first, leftover = value.first_drop_amount, value.leftover_after_first_drop
    if not all(_quantity_is_available(item) for item in (residual, first, leftover)):
        return
    expected_first = min(int(residual.value), OJAMA_MAX_DROP_PER_TURN)
    if first.value != expected_first or leftover.value != int(residual.value) - expected_first:
        raise CanonicalObservationError("residualとfirst drop/leftoverが一致しません")


def _validate_active_chain_side(value: CausalLedgerSnapshot) -> None:
    p1, p2 = value.p1.chain_active, value.p2.chain_active
    if not _both_available(p1, p2):
        return
    active = [side for side, flag in (("p1", p1.value), ("p2", p2.value)) if flag]
    expected = None if not active else (active[0] if len(active) == 1 else "both")
    _validate_derived_side(value.active_chain_side, expected, "active chain side")


def _validate_recipient(value: CausalLedgerSnapshot) -> None:
    p1, p2 = value.p1.post_cancel_residual, value.p2.post_cancel_residual
    if not _both_available(p1, p2):
        return
    recipients = [side for side, amount in (("p1", p1.value), ("p2", p2.value)) if amount]
    if len(recipients) > 1:
        raise CanonicalObservationError("相殺後residualを両sideへ同時に残せません")
    expected = recipients[0] if recipients else None
    _validate_derived_side(value.recipient, expected, "recipient")


def _validate_derived_side(value: AvailableSide, expected: str | None, label: str) -> None:
    expected_state = AvailabilityState.KNOWN if expected else AvailabilityState.KNOWN_ZERO
    if value.availability != expected_state or value.value != expected:
        raise CanonicalObservationError(f"{label}がside別物理状態と一致しません")


def _quantity_is_available(value: AvailableInt) -> bool:
    return value.availability in {AvailabilityState.KNOWN, AvailabilityState.KNOWN_ZERO}


def _both_available(first: AvailableBool | AvailableInt, second: AvailableBool | AvailableInt) -> bool:
    states = {AvailabilityState.KNOWN, AvailabilityState.KNOWN_ZERO}
    return first.availability in states and second.availability in states


def _validate_observation_identity(value: CanonicalObservationV2) -> None:
    if value.schema_version != CANONICAL_OBSERVATION_SCHEMA_VERSION:
        raise CanonicalObservationError("canonical observation schema versionが不正です")
    for item, label in ((value.observation_id, "observation ID"),
                        (value.source_video_id, "source video ID"),
                        (value.build_id, "build ID")):
        _require_text(item, label)
    for item, label in ((value.game_idx, "game index"),
                        (value.through_event_seq, "through event sequence"),
                        (value.cutoff_frame, "cutoff frame"), (value.cutoff_ms, "cutoff ms"),
                        (value.boundary_segment, "boundary segment")):
        _require_nonnegative_int(item, label)


def _validate_observation_types(value: CanonicalObservationV2) -> None:
    expected = (CanonicalSideObservation, CanonicalSideObservation, CausalLedgerSnapshot,
                ObservationProvenance, ObservationQuality)
    actual = (value.p1, value.p2, value.ledger, value.provenance, value.quality)
    if any(not isinstance(item, kind) for item, kind in zip(actual, expected, strict=True)):
        raise CanonicalObservationError("canonical observationのnested型が不正です")
    if value.ledger.ledger_prefix_digest != value.provenance.event_prefix_semantic_digest:
        raise CanonicalObservationError("ledgerとobservationのevent prefix digestが不一致です")


def _validate_causal_cutoff(value: CanonicalObservationV2) -> None:
    for provenance in _field_provenances(value):
        if provenance.available_frame is None:
            continue
        if provenance.available_frame > value.cutoff_frame or provenance.available_ms > value.cutoff_ms:
            raise CanonicalObservationError("cutoffより未来の値をcanonical observationへ入れられません")


def _validate_observation_quality(value: CanonicalObservationV2) -> None:
    states = _availability_states(value)
    if (AvailabilityState.INTEGRITY_FAULT in states
            and value.quality.status != ObservationStatus.INTEGRITY_FAULT):
        raise CanonicalObservationError("field integrity faultをqualityが公開していません")
    if (AvailabilityState.UNSUPPORTED in states
            and value.quality.status == ObservationStatus.READY):
        raise CanonicalObservationError("unsupported fieldをREADYとして公開できません")
    boards_known = all(side.board.availability == AvailabilityState.KNOWN
                       for side in (value.p1, value.p2))
    if not boards_known and value.quality.status == ObservationStatus.READY:
        raise CanonicalObservationError("両側STABLE盤面なしではREADYにできません")


def _field_provenances(value: CanonicalObservationV2) -> tuple[FieldProvenance, ...]:
    result: list[FieldProvenance] = [value.ledger.provenance]
    for side in (value.p1, value.p2):
        result.append(side.board.provenance)
        result.extend(item.provenance for item in _side_available_values(side))
    for side in (value.ledger.p1, value.ledger.p2):
        result.extend(getattr(side, item.name).provenance for item in fields(side))
    result.extend((value.ledger.active_chain_side.provenance,
                   value.ledger.recipient.provenance))
    return tuple(result)


def _side_available_values(side: CanonicalSideObservation) -> tuple[Any, ...]:
    pairs = (side.pieces.current, side.pieces.next, side.pieces.double_next)
    colors = tuple(color for pair in pairs for color in (pair.axis, pair.child))
    return (*colors, side.all_clear_pending, side.raw_score, side.tsumo_count)


def _availability_states(value: CanonicalObservationV2) -> set[AvailabilityState]:
    states = {side.board.availability for side in (value.p1, value.p2)}
    for side in (value.p1, value.p2):
        states.update(item.availability for item in _side_available_values(side))
    for side in (value.ledger.p1, value.ledger.p2):
        states.update(getattr(side, item.name).availability for item in fields(side))
    states.update((value.ledger.active_chain_side.availability,
                   value.ledger.recipient.availability))
    return states


def _nested_validatables(value: CanonicalObservationV2) -> tuple[Any, ...]:
    nested: list[Any] = [value.p1, value.p2, value.ledger, value.provenance, value.quality]
    for side in (value.p1, value.p2):
        nested.extend((side.board, side.pieces, side.all_clear_pending,
                       side.raw_score, side.tsumo_count, side.board.provenance))
        for pair in (side.pieces.current, side.pieces.next, side.pieces.double_next):
            nested.append(pair)
            nested.extend((pair.axis, pair.child, pair.axis.provenance, pair.child.provenance))
    _append_ledger_validatables(nested, value.ledger)
    return tuple(nested)


def _append_ledger_validatables(items: list[Any], ledger: CausalLedgerSnapshot) -> None:
    items.extend((ledger.p1, ledger.p2, ledger.active_chain_side,
                  ledger.recipient, ledger.provenance))
    for side in (ledger.p1, ledger.p2):
        for item in fields(side):
            available = getattr(side, item.name)
            items.extend((available, available.provenance))


def _observation_from_dict(value: Mapping[str, Any]) -> CanonicalObservationV2:
    data = _exact_mapping(value, _field_names(CanonicalObservationV2), "observation")
    return CanonicalObservationV2(
        observation_id=_text(data["observation_id"], "observation ID"),
        source_video_id=_text(data["source_video_id"], "source video ID"),
        build_id=_text(data["build_id"], "build ID"), game_idx=data["game_idx"],
        through_event_seq=data["through_event_seq"], cutoff_frame=data["cutoff_frame"],
        cutoff_ms=data["cutoff_ms"], boundary_segment=data["boundary_segment"],
        p1=_side_from_dict(data["p1"]), p2=_side_from_dict(data["p2"]),
        ledger=_ledger_from_dict(data["ledger"]),
        provenance=_observation_provenance_from_dict(data["provenance"]),
        quality=_quality_from_dict(data["quality"]), schema_version=data["schema_version"],
    )


def _side_from_dict(value: Any) -> CanonicalSideObservation:
    data = _exact_mapping(value, _field_names(CanonicalSideObservation), "side observation")
    return CanonicalSideObservation(
        board=_board_from_dict(data["board"]), pieces=_queue_from_dict(data["pieces"]),
        all_clear_pending=_available_from_dict(data["all_clear_pending"], AvailableBool),
        raw_score=_available_from_dict(data["raw_score"], AvailableInt),
        tsumo_count=_available_from_dict(data["tsumo_count"], AvailableInt),
    )


def _board_from_dict(value: Any) -> StableBoardObservation:
    data = _exact_mapping(value, _field_names(StableBoardObservation), "board")
    raw_grid, raw_mask = data["grid"], data["known_mask"]
    grid = None if raw_grid is None else _matrix_from_json(raw_grid, "grid")
    mask = None if raw_mask is None else _matrix_from_json(raw_mask, "mask")
    return StableBoardObservation(grid, mask, _availability(data["availability"]),
                                  _field_provenance_from_dict(data["provenance"]))


def _queue_from_dict(value: Any) -> PieceQueueObservation:
    data = _exact_mapping(value, _field_names(PieceQueueObservation), "piece queue")
    return PieceQueueObservation(*(_pair_from_dict(data[name])
                                   for name in ("current", "next", "double_next")))


def _pair_from_dict(value: Any) -> PiecePairObservation:
    data = _exact_mapping(value, _field_names(PiecePairObservation), "piece pair")
    return PiecePairObservation(
        _available_from_dict(data["axis"], AvailableColor),
        _available_from_dict(data["child"], AvailableColor),
    )


_AvailableT = TypeVar("_AvailableT", AvailableInt, AvailableBool, AvailableColor, AvailableSide)
_EnumT = TypeVar("_EnumT", bound=StrEnum)


def _available_from_dict(value: Any, kind: type[_AvailableT]) -> _AvailableT:
    data = _exact_mapping(value, _field_names(kind), kind.__name__)
    return kind(data["value"], _availability(data["availability"]),
                _field_provenance_from_dict(data["provenance"]))


def _ledger_side_from_dict(value: Any) -> CausalLedgerSideSnapshot:
    data = _exact_mapping(value, _field_names(CausalLedgerSideSnapshot), "ledger side")
    bool_names = {"chain_active", "send_waiting", "placement_waiting",
                  "chain_end_waiting", "garbage_falling"}
    parsed = {name: _available_from_dict(raw, AvailableBool if name in bool_names else AvailableInt)
              for name, raw in data.items()}
    return CausalLedgerSideSnapshot(**parsed)


def _ledger_from_dict(value: Any) -> CausalLedgerSnapshot:
    data = _exact_mapping(value, _field_names(CausalLedgerSnapshot), "ledger")
    return CausalLedgerSnapshot(
        p1=_ledger_side_from_dict(data["p1"]), p2=_ledger_side_from_dict(data["p2"]),
        active_chain_side=_available_from_dict(data["active_chain_side"], AvailableSide),
        recipient=_available_from_dict(data["recipient"], AvailableSide),
        ledger_prefix_digest=data["ledger_prefix_digest"],
        provenance=_field_provenance_from_dict(data["provenance"]),
    )


def _field_provenance_from_dict(value: Any) -> FieldProvenance:
    data = _exact_mapping(value, _field_names(FieldProvenance), "field provenance")
    return FieldProvenance(
        kind=_enum(ProvenanceKind, data["kind"]),
        source_event_ids=tuple(_list(data["source_event_ids"], "source event IDs")),
        available_frame=data["available_frame"], available_ms=data["available_ms"],
        confidence_milli=data["confidence_milli"],
        reason_codes=tuple(_list(data["reason_codes"], "reason codes")),
    )


def _observation_provenance_from_dict(value: Any) -> ObservationProvenance:
    data = _exact_mapping(value, _field_names(ObservationProvenance), "observation provenance")
    return ObservationProvenance(**data)


def _quality_from_dict(value: Any) -> ObservationQuality:
    data = _exact_mapping(value, _field_names(ObservationQuality), "quality")
    return ObservationQuality(
        _enum(ObservationStatus, data["status"]),
        tuple(_list(data["reason_codes"], "quality reason codes")), data["quarantined"],
    )


def _exact_mapping(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CanonicalObservationError(f"{label}のkey集合がschemaと一致しません")
    return value


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


def _field_names(kind: type[Any]) -> set[str]:
    return {item.name for item in fields(kind)}


def _availability(value: Any) -> AvailabilityState:
    return _enum(AvailabilityState, value)


def _enum(kind: type[_EnumT], value: Any) -> _EnumT:
    try:
        return kind(value)
    except (TypeError, ValueError) as error:
        raise CanonicalObservationError(f"{kind.__name__}の値が不正です") from error


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CanonicalObservationError(f"{label}はJSON array必須です")
    return value


def _matrix_from_json(value: Any, label: str) -> tuple[tuple[Any, ...], ...]:
    rows = _list(value, label)
    return tuple(tuple(_list(row, f"{label} row")) for row in rows)


def _text(value: Any, label: str) -> str:
    _require_text(value, label)
    return value


def _validate_optional_position(frame: int | None, milliseconds: int | None) -> None:
    if (frame is None) != (milliseconds is None):
        raise CanonicalObservationError("available frame/msは対で指定します")
    if frame is not None:
        _require_nonnegative_int(frame, "available frame")
        _require_nonnegative_int(milliseconds, "available ms")


def _validate_provenance_shape(value: FieldProvenance) -> None:
    known = value.kind in _KNOWN_PROVENANCE_KINDS
    if known and (not value.source_event_ids or value.available_frame is None):
        raise CanonicalObservationError("known provenanceにevent/timingがありません")
    if known and value.reason_codes:
        raise CanonicalObservationError("known provenanceに欠測理由は指定できません")
    if not known and not value.reason_codes:
        raise CanonicalObservationError("非可用provenanceには理由が必要です")
    has_events = bool(value.source_event_ids)
    has_timing = value.available_frame is not None
    if has_events != has_timing or (value.confidence_milli is not None and not has_timing):
        raise CanonicalObservationError("provenanceのevent/timing/confidenceが不整合です")


def _require_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CanonicalObservationError(f"{label}は空白なしの非空文字列必須です")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise CanonicalObservationError(f"{label}に制御文字は使えません")


def _require_text_tuple(value: object, label: str, *, allow_empty: bool) -> None:
    if type(value) is not tuple or (not allow_empty and not value):
        raise CanonicalObservationError(f"{label}はtuple必須です")
    for item in value:
        _require_text(item, label)


def _require_sorted_unique(value: tuple[str, ...], label: str) -> None:
    if tuple(sorted(set(value))) != value:
        raise CanonicalObservationError(f"{label}は昇順かつ重複なし必須です")


def _require_nonnegative_int(value: object, label: str) -> None:
    if type(value) is not int or value < 0:
        raise CanonicalObservationError(f"{label}は非負整数必須です")


def _require_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise CanonicalObservationError(f"{label}は小文字64桁SHA-256必須です")


def _is_int_between(value: object, minimum: int, maximum: int) -> bool:
    return type(value) is int and minimum <= value <= maximum


def _opponent(side: str) -> str:
    return "p2" if side == "p1" else "p1"


__all__ = [
    "ACTIVE_COLORS", "AvailabilityState", "AvailableBool", "AvailableColor",
    "AvailableInt", "AvailableSide", "CANONICAL_OBSERVATION_SCHEMA_VERSION",
    "CanonicalObservationError", "CanonicalObservationV2", "CanonicalSideObservation",
    "CausalLedgerSideSnapshot", "CausalLedgerSnapshot", "FieldProvenance",
    "ImmutableGrid", "ImmutableMask", "ObservationProvenance", "ObservationQuality",
    "ObservationStatus", "PiecePairObservation", "PieceQueueObservation",
    "ProvenanceKind", "StableBoardObservation", "canonical_json_bytes",
    "validate_canonical_observation",
]
