"""確定event prefixを学習・本番共通のcanonical観測へ変換する。"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from src.canonical_observation_v2 import (
    AvailabilityState,
    AvailableBool,
    AvailableColor,
    AvailableInt,
    AvailableSide,
    CanonicalObservationV2,
    CanonicalSideObservation,
    CausalLedgerSideSnapshot,
    CausalLedgerSnapshot,
    FieldProvenance,
    ObservationProvenance,
    ObservationQuality,
    ObservationStatus,
    PiecePairObservation,
    PieceQueueObservation,
    ProvenanceKind,
    StableBoardObservation,
)
from src.event_causal_exchange_v1 import (
    CausalExchangeReplay,
    settled_pending_disagreement,
)
from src.event_learning_tables_v1 import KNOWN_EVENT_TYPES, POSTHOC_EVENT_TYPES
from src.event_source_v1 import (
    CommittedBatch,
    SemanticEventHasher,
    semantic_events_sha256,
    validate_event_batch,
)
from src.scoring import OJAMA_MAX_DROP_PER_TURN


CANONICAL_ADAPTER_VERSION = "canonical-observation-adapter/v2"
SIDES = ("p1", "p2")
BOARD_EVENT_TYPE = "stable_board_observed"
BOUNDARY_EVENT_TYPE = "match_boundary_evidence"
FuturePolicy = Literal["reject", "exclude"]


class CanonicalObservationAdapterError(ValueError):
    """因果prefixをcanonical観測へ安全に変換できない。"""


@dataclass(frozen=True, slots=True)
class CanonicalCutoff:
    """予測時点で利用可能な最大フレームと最大ミリ秒。"""

    available_frame: int
    available_ms: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.available_frame, "cutoff frame")
        _require_nonnegative_int(self.available_ms, "cutoff ms")


@dataclass(frozen=True, slots=True)
class _PrefixIdentity:
    source_video_id: str
    build_id: str
    attempt_id: str


@dataclass(frozen=True, slots=True)
class _SelectedPrefix:
    batches: tuple[CommittedBatch, ...]
    identity: _PrefixIdentity | None
    digest: str
    through_sequence: int


@dataclass(slots=True)
class _ReplayState:
    """canonical DTOへ写す前の、prefixだけから決まる内部状態。"""

    boards: dict[str, dict[str, Any]] = field(default_factory=dict)
    chain_active: dict[str, bool] = field(
        default_factory=lambda: {side: False for side in SIDES}
    )
    provisional: dict[str, dict[str, int | None] | None] = field(
        default_factory=lambda: {side: None for side in SIDES}
    )
    all_clear: dict[str, tuple[bool, dict[str, Any]] | None] = field(
        default_factory=lambda: {side: None for side in SIDES}
    )
    causal: CausalExchangeReplay = field(default_factory=CausalExchangeReplay)
    physical_pending: dict[str, int] = field(
        default_factory=lambda: {side: 0 for side in SIDES}
    )
    pending_disagreement: int = 0
    boundary_segment: int = 0
    unsupported_reasons: set[str] = field(default_factory=set)
    accepted_event_ids: list[str] = field(default_factory=list)
    last_event: dict[str, Any] | None = None

    def reset_for_boundary(self) -> None:
        self.boards.clear()
        self.chain_active = {side: False for side in SIDES}
        self.provisional = {side: None for side in SIDES}
        self.all_clear = {side: None for side in SIDES}
        self.causal.reset()
        self.physical_pending = {side: 0 for side in SIDES}
        self.pending_disagreement = 0
        self.unsupported_reasons.clear()
        self.accepted_event_ids.clear()
        self.last_event = None
        self.boundary_segment += 1


def canonical_observation_from_committed_prefix(
    batches: Iterable[CommittedBatch],
    cutoff: CanonicalCutoff,
    *,
    future_policy: FuturePolicy = "reject",
) -> CanonicalObservationV2 | None:
    """commit済みprefixだけからcanonical観測を決定論的に作る。"""

    selected = _select_prefix(tuple(batches), cutoff, future_policy)
    if not selected.batches or selected.identity is None:
        return None
    replay = _replay(selected.batches)
    if set(replay.boards) != set(SIDES):
        return None
    return _build_canonical_observation(replay, selected, cutoff)


def iter_canonical_observations(
    batches: Iterable[CommittedBatch],
) -> Iterator[CanonicalObservationV2]:
    """event列を一度だけreplayし、各因果batch後のcanonical観測を返す。"""

    replay = _ReplayState()
    hasher = SemanticEventHasher()
    identity: _PrefixIdentity | None = None
    expected_seq, previous_available = 0, (-1, -1)
    for batch in batches:
        identity, expected_seq, previous_available = _validate_batch(
            batch, identity, expected_seq, previous_available,
        )
        if _contains_posthoc_event(batch):
            break
        _reject_noncausal_events(batch)
        for event in batch.events:
            hasher.add(event)
            if event["event_type"] == BOUNDARY_EVENT_TYPE:
                replay.reset_for_boundary()
            else:
                _apply_event(replay, event)
        _refresh_pending_integrity(replay)
        if set(replay.boards) == set(SIDES) and replay.last_event is not None:
            frame, milliseconds = _available_position(batch)
            selected = _SelectedPrefix(
                (), identity, hasher.hexdigest(), int(batch.events[-1]["seq"]),
            )
            yield _build_canonical_observation(
                replay, selected, CanonicalCutoff(frame, milliseconds),
            )


def _contains_posthoc_event(batch: CommittedBatch) -> bool:
    return any(str(event["event_type"]) in POSTHOC_EVENT_TYPES for event in batch.events)


def _select_prefix(
    batches: Sequence[CommittedBatch], cutoff: CanonicalCutoff,
    future_policy: FuturePolicy,
) -> _SelectedPrefix:
    if future_policy not in {"reject", "exclude"}:
        raise CanonicalObservationAdapterError("future policyが不正です")
    hasher = SemanticEventHasher()
    accepted: list[CommittedBatch] = []
    identity: _PrefixIdentity | None = None
    expected_seq = 0
    previous_available = (-1, -1)
    for batch in batches:
        identity, expected_seq, previous_available = _validate_batch(
            batch, identity, expected_seq, previous_available,
        )
        if not _at_or_before_cutoff(batch, cutoff):
            if future_policy == "reject":
                raise CanonicalObservationAdapterError("cutoff後のevent batchが混入しています")
            continue
        _reject_noncausal_events(batch)
        accepted.append(copy.deepcopy(batch))
        for event in batch.events:
            hasher.add(event)
    through = -1 if not accepted else int(accepted[-1].events[-1]["seq"])
    return _SelectedPrefix(tuple(accepted), identity, hasher.hexdigest(), through)


def _validate_batch(
    batch: CommittedBatch, identity: _PrefixIdentity | None,
    expected_seq: int, previous_available: tuple[int, int],
) -> tuple[_PrefixIdentity, int, tuple[int, int]]:
    if not isinstance(batch, CommittedBatch) or not batch.events:
        raise CanonicalObservationAdapterError("committed batch以外は入力できません")
    try:
        validate_event_batch(batch.events, expected_first_seq=expected_seq)
    except ValueError as error:
        raise CanonicalObservationAdapterError(f"event batch検証失敗: {error}") from error
    if batch.semantic_sha256 != semantic_events_sha256(batch.events):
        raise CanonicalObservationAdapterError("batch semantic hashが一致しません")
    if batch.batch_id != batch.events[0]["availability_batch_id"]:
        raise CanonicalObservationAdapterError("commitとavailability batch IDが一致しません")
    current = _event_identity(batch.events[0])
    if identity is not None and current != identity:
        raise CanonicalObservationAdapterError("event prefixのsource identityが変化しました")
    available = _available_position(batch)
    if available[0] < previous_available[0] or available[1] < previous_available[1]:
        raise CanonicalObservationAdapterError("event prefixの利用可能時刻が巻き戻っています")
    return current, int(batch.events[-1]["seq"]) + 1, available


def _event_identity(event: Mapping[str, Any]) -> _PrefixIdentity:
    values = tuple(str(event.get(name, "")) for name in (
        "source_video_id", "build_id", "attempt_id",
    ))
    if any(not value for value in values):
        raise CanonicalObservationAdapterError("event identityが空です")
    return _PrefixIdentity(*values)


def _available_position(batch: CommittedBatch) -> tuple[int, int]:
    timing = batch.events[0]["timing"]
    return int(timing["available_frame"]), int(timing["available_ms"])


def _at_or_before_cutoff(batch: CommittedBatch, cutoff: CanonicalCutoff) -> bool:
    frame, milliseconds = _available_position(batch)
    return frame <= cutoff.available_frame and milliseconds <= cutoff.available_ms


def _reject_noncausal_events(batch: CommittedBatch) -> None:
    event_types = {str(event["event_type"]) for event in batch.events}
    posthoc = sorted(event_types & POSTHOC_EVENT_TYPES)
    if posthoc:
        raise CanonicalObservationAdapterError(f"posthoc eventは入力禁止です: {posthoc}")
    unknown = sorted(event_types - KNOWN_EVENT_TYPES)
    if unknown:
        raise CanonicalObservationAdapterError(f"未登録eventは入力禁止です: {unknown}")


def _replay(batches: Sequence[CommittedBatch]) -> _ReplayState:
    replay = _ReplayState()
    for batch in batches:
        for event in batch.events:
            if event["event_type"] == BOUNDARY_EVENT_TYPE:
                replay.reset_for_boundary()
                continue
            _apply_event(replay, event)
        _refresh_pending_integrity(replay)
    return replay


def _apply_event(replay: _ReplayState, event: Mapping[str, Any]) -> None:
    event_type = str(event["event_type"])
    replay.causal.observe(event)
    _apply_physical_pending(replay, event)
    replay.accepted_event_ids.append(str(event["event_id"]))
    replay.last_event = copy.deepcopy(dict(event))
    if event.get("payload", {}).get("model_input_allowed") is False:
        replay.unsupported_reasons.add("event_explicitly_disallowed")
    if event_type == BOARD_EVENT_TYPE:
        _apply_board(replay, event)
    elif event_type == "attack_provisional_updated":
        _apply_provisional(replay, event)
    elif event_type in {"attack_finalized", "attack_provisional_unfinalized"}:
        _clear_chain(replay, _side(event))
    elif event_type == "chain_started":
        replay.chain_active[_side(event)] = True
    elif event_type == "all_clear_gained":
        replay.all_clear[_side(event)] = (True, copy.deepcopy(dict(event)))
    elif event_type == "all_clear_consumed":
        replay.all_clear[_side(event)] = (False, copy.deepcopy(dict(event)))
    elif event_type == "garbage_same_frame_order_ambiguous":
        replay.unsupported_reasons.add("same_frame_accounting_order_ambiguous")


def _apply_physical_pending(replay: _ReplayState, event: Mapping[str, Any]) -> None:
    event_type = str(event["event_type"])
    payload = event.get("payload", {})
    if event_type == "garbage_sent":
        recipient = payload.get("recipient")
        if recipient not in SIDES:
            raise CanonicalObservationAdapterError("garbage sentのrecipientが不正です")
        replay.physical_pending[str(recipient)] += _event_amount(payload, "sent_amount")
    elif event_type in {"garbage_cancelled", "garbage_fall_completed"}:
        key = "cancelled_amount" if event_type == "garbage_cancelled" else "modeled_landed_amount"
        _decrease_physical_pending(replay, _side(event), _event_amount(payload, key))
    elif event_type == "garbage_same_frame_order_ambiguous":
        _decrease_physical_pending(
            replay, _side(event), _event_amount(payload, "ambiguous_amount"),
        )
    elif event_type == "garbage_expired_at_boundary":
        side, amount = _side(event), _event_amount(payload, "expired_amount")
        replay.physical_pending[side] = max(0, replay.physical_pending[side] - amount)


def _event_amount(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int or value < 0:
        raise CanonicalObservationAdapterError(f"{key}が非負整数ではありません")
    return value


def _decrease_physical_pending(replay: _ReplayState, side: str, amount: int) -> None:
    if amount > replay.physical_pending[side]:
        raise CanonicalObservationAdapterError("物理会計の未解決量が負になります")
    replay.physical_pending[side] -= amount


def _refresh_pending_integrity(replay: _ReplayState) -> None:
    replay.pending_disagreement = settled_pending_disagreement(
        replay.physical_pending,
        {side: replay.causal.pending(side) for side in SIDES},
        replay.chain_active,
        {
            side: None if replay.provisional[side] is None
            else replay.provisional[side].get("generated_amount")
            for side in SIDES
        },
        causal_usable=replay.causal.usable,
    )


def _apply_board(replay: _ReplayState, event: Mapping[str, Any]) -> None:
    assertion = event.get("assertion")
    payload = event.get("payload")
    if assertion != {"state": "confirmed", "value_form": "exact"}:
        raise CanonicalObservationAdapterError("STABLE盤面がconfirmed/exactではありません")
    if not isinstance(payload, Mapping) or payload.get("board_provenance") != "observed":
        raise CanonicalObservationAdapterError("STABLE confirmed_boardが実観測ではありません")
    _validate_grid_and_mask(payload)
    side = _side(event)
    replay.boards[side] = copy.deepcopy(dict(event))
    replay.chain_active[side] = False
    context = payload.get("observed_context", {})
    if not isinstance(context, Mapping):
        raise CanonicalObservationAdapterError("observed contextがmappingではありません")
    value = context.get("all_clear_pending")
    replay.all_clear[side] = (
        (value, copy.deepcopy(dict(event))) if type(value) is bool else None
    )


def _validate_grid_and_mask(payload: Mapping[str, Any]) -> None:
    grid = payload.get("grid")
    mask = payload.get("unknown_mask")
    if not _is_matrix(grid, 13, 6):
        raise CanonicalObservationAdapterError("confirmed_boardが13x6ではありません")
    if not _is_matrix(mask, 13, 6):
        raise CanonicalObservationAdapterError("unknown maskが13x6ではありません")
    if any(type(value) is not int or value not in {0, 1, 2, 3, 4, 5, 9, 10}
           for row in grid for value in row):
        raise CanonicalObservationAdapterError("confirmed_boardに不正なcellがあります")
    if any(value not in {0, 1, False, True} for row in mask for value in row):
        raise CanonicalObservationAdapterError("unknown maskに不正な値があります")
    expected = [[cell == 10 for cell in row] for row in grid]
    if [[bool(cell) for cell in row] for row in mask] != expected:
        raise CanonicalObservationAdapterError("COLOR_UNKNOWNとunknown maskが一致しません")


def _is_matrix(value: Any, rows: int, columns: int) -> bool:
    return bool(
        isinstance(value, list) and len(value) == rows
        and all(isinstance(row, list) and len(row) == columns for row in value)
    )


def _apply_provisional(replay: _ReplayState, event: Mapping[str, Any]) -> None:
    side = _side(event)
    payload = event.get("payload", {})
    replay.provisional[side] = {
        "generated_amount": _optional_nonnegative_int(
            payload.get("provisional_generated_amount"), "provisional generated amount",
            required=True,
        ),
        "score": _optional_nonnegative_int(payload.get("provisional_score"), "provisional score"),
        "chain_count": _optional_nonnegative_int(payload.get("chain_count"), "chain count"),
    }
    replay.chain_active[side] = True


def _optional_nonnegative_int(
    value: Any, label: str, *, required: bool = False,
) -> int | None:
    if value is None and not required:
        return None
    if type(value) is not int or value < 0:
        raise CanonicalObservationAdapterError(f"{label}が非負整数ではありません")
    return value


def _clear_chain(replay: _ReplayState, side: str) -> None:
    replay.provisional[side] = None
    replay.chain_active[side] = False


def _side(event: Mapping[str, Any]) -> str:
    side = event.get("side")
    if side not in SIDES:
        raise CanonicalObservationAdapterError("sideがp1/p2ではありません")
    return str(side)


def _require_nonnegative_int(value: Any, label: str) -> None:
    if type(value) is not int or value < 0:
        raise CanonicalObservationAdapterError(f"{label}は0以上の整数必須です")


def _build_canonical_observation(
    replay: _ReplayState, selected: _SelectedPrefix, cutoff: CanonicalCutoff,
) -> CanonicalObservationV2:
    """prefixから一意の公開DTOを構築する。"""

    identity = selected.identity
    if identity is None or replay.last_event is None:
        raise CanonicalObservationAdapterError("canonical観測のidentityがありません")
    ledger = _build_ledger(replay, selected.digest)
    quality = _build_quality(replay)
    return CanonicalObservationV2(
        observation_id=f"{identity.build_id}:canonical-after-{selected.through_sequence:012d}",
        source_video_id=identity.source_video_id,
        build_id=identity.build_id,
        game_idx=replay.boundary_segment,
        through_event_seq=selected.through_sequence,
        cutoff_frame=cutoff.available_frame,
        cutoff_ms=cutoff.available_ms,
        boundary_segment=replay.boundary_segment,
        p1=_build_side(replay, "p1"),
        p2=_build_side(replay, "p2"),
        ledger=ledger,
        provenance=ObservationProvenance(
            event_prefix_semantic_digest=selected.digest,
            source_group_id=identity.source_video_id,
            attempt_id=identity.attempt_id,
        ),
        quality=quality,
    )


def _build_side(replay: _ReplayState, side: str) -> CanonicalSideObservation:
    event = replay.boards[side]
    payload = event["payload"]
    context = payload.get("observed_context", {})
    board_provenance = _event_provenance(event, ProvenanceKind.DIRECT_OBSERVATION)
    grid = tuple(tuple(int(cell) for cell in row) for row in payload["grid"])
    known_mask = tuple(
        tuple(not bool(cell) for cell in row) for row in payload["unknown_mask"]
    )
    return CanonicalSideObservation(
        board=StableBoardObservation.known(grid, known_mask, board_provenance),
        pieces=_piece_queue(context, event),
        all_clear_pending=_all_clear_value(replay.all_clear[side]),
        raw_score=_context_int(context, "score", event),
        tsumo_count=_context_int(context, "tsumo_count", event),
    )


def _piece_queue(
    context: Mapping[str, Any], event: Mapping[str, Any],
) -> PieceQueueObservation:
    return PieceQueueObservation(
        current=_piece_pair(context, "current_pair", event),
        next=_piece_pair(context, "next_pair", event),
        double_next=_piece_pair(context, "double_next_pair", event),
    )


def _piece_pair(
    context: Mapping[str, Any], key: str, event: Mapping[str, Any],
) -> PiecePairObservation:
    value = context.get(key)
    if value is None:
        unavailable = _unknown_color(f"{key}_not_observed")
        return PiecePairObservation(unavailable, unavailable)
    if not isinstance(value, Mapping):
        raise CanonicalObservationAdapterError(f"{key}がmappingではありません")
    provenance = _event_provenance(event, ProvenanceKind.DIRECT_OBSERVATION)
    return PiecePairObservation(
        _available_color(value.get("first"), provenance, f"{key}_first"),
        _available_color(value.get("second"), provenance, f"{key}_second"),
    )


def _available_color(
    value: Any, provenance: FieldProvenance, label: str,
) -> AvailableColor:
    if value is None:
        return _unknown_color(f"{label}_not_observed")
    if type(value) is not int or value not in {1, 2, 3, 4, 5}:
        raise CanonicalObservationAdapterError(f"{label}の色が1..5ではありません")
    return AvailableColor.known(value, provenance)


def _unknown_color(reason: str) -> AvailableColor:
    provenance = FieldProvenance.unavailable(AvailabilityState.UNKNOWN, (reason,))
    return AvailableColor.unavailable(AvailabilityState.UNKNOWN, provenance)


def _all_clear_value(
    entry: tuple[bool, dict[str, Any]] | None,
) -> AvailableBool:
    if entry is None:
        provenance = FieldProvenance.unavailable(
            AvailabilityState.UNKNOWN, ("all_clear_state_not_observed",),
        )
        return AvailableBool.unavailable(AvailabilityState.UNKNOWN, provenance)
    value, event = entry
    return AvailableBool.known(
        value, _event_provenance(event, ProvenanceKind.DIRECT_OBSERVATION),
    )


def _context_int(
    context: Mapping[str, Any], key: str, event: Mapping[str, Any],
) -> AvailableInt:
    value = context.get(key)
    if value is None:
        provenance = FieldProvenance.unavailable(
            AvailabilityState.UNKNOWN, (f"{key}_not_observed",),
        )
        return AvailableInt.unavailable(AvailabilityState.UNKNOWN, provenance)
    parsed = _optional_nonnegative_int(value, key, required=True)
    return AvailableInt.known(
        int(parsed), _event_provenance(event, ProvenanceKind.DIRECT_OBSERVATION),
    )


def _build_ledger(replay: _ReplayState, digest: str) -> CausalLedgerSnapshot:
    provenance = _ledger_provenance(replay)
    p1 = _build_ledger_side(replay, "p1", provenance)
    p2 = _build_ledger_side(replay, "p2", provenance)
    return CausalLedgerSnapshot(
        p1=p1,
        p2=p2,
        active_chain_side=_active_chain_side(replay, provenance),
        recipient=_recipient(replay, provenance),
        ledger_prefix_digest=digest,
        provenance=provenance,
    )


def _build_ledger_side(
    replay: _ReplayState, side: str, provenance: FieldProvenance,
) -> CausalLedgerSideSnapshot:
    residual = _residual_amount(replay, side)
    provisional = replay.provisional[side]
    return CausalLedgerSideSnapshot(
        pending_garbage=_causal_int(replay, replay.causal.pending(side), provenance),
        effective_rate=_optional_ledger_int(replay.causal.rate(side), provenance),
        chain_active=AvailableBool.known(replay.chain_active[side], provenance),
        chain_step=_chain_quantity(replay, side, "chain_count", provenance),
        provisional_generated=_chain_quantity(replay, side, "generated_amount", provenance),
        provisional_score=_chain_quantity(replay, side, "score", provenance),
        provisional_chain_count=_chain_quantity(replay, side, "chain_count", provenance),
        finalized_unsettled_attack=_causal_int(replay, replay.causal.pending(side), provenance),
        post_cancel_residual=_causal_int(replay, residual, provenance),
        send_waiting=_unknown_bool("send_waiting_not_observed"),
        placement_waiting=_unknown_bool("placement_waiting_not_observed"),
        chain_end_waiting=_unknown_bool("chain_end_waiting_not_observed"),
        garbage_falling=_unknown_bool("garbage_falling_not_observed"),
        first_drop_amount=_first_drop(replay, residual, provenance),
        leftover_after_first_drop=_leftover(replay, residual, provenance),
    )


def _chain_quantity(
    replay: _ReplayState, side: str, key: str, provenance: FieldProvenance,
) -> AvailableInt:
    if not replay.chain_active[side]:
        return AvailableInt.known(0, provenance)
    provisional = replay.provisional[side]
    value = None if provisional is None else provisional.get(key)
    return _optional_ledger_int(value, provenance, reason=f"{key}_not_observed")


def _residual_amount(replay: _ReplayState, side: str) -> int | None:
    balance = replay.causal.observed_balance()
    if balance is None:
        return None
    return max(0, -balance) if side == "p1" else max(0, balance)


def _causal_int(
    replay: _ReplayState, value: int | None, provenance: FieldProvenance,
) -> AvailableInt:
    if value is not None and _causal_ready(replay):
        return AvailableInt.known(value, provenance)
    reasons = _causal_reason_codes(replay)
    unavailable = FieldProvenance.unavailable(AvailabilityState.UNSUPPORTED, reasons)
    return AvailableInt.unavailable(AvailabilityState.UNSUPPORTED, unavailable)


def _causal_ready(replay: _ReplayState) -> bool:
    return replay.causal.usable and replay.pending_disagreement == 0


def _causal_reason_codes(replay: _ReplayState) -> tuple[str, ...]:
    reasons = list(replay.causal.reason_codes)
    if replay.pending_disagreement:
        reasons.append("causal_ledger_pending_disagrees_while_settled")
    return tuple(reasons) or ("causal_ledger_unavailable",)


def _optional_ledger_int(
    value: int | None, provenance: FieldProvenance, *, reason: str = "value_not_observed",
) -> AvailableInt:
    if value is not None:
        return AvailableInt.known(value, provenance)
    unavailable = FieldProvenance.unavailable(AvailabilityState.UNKNOWN, (reason,))
    return AvailableInt.unavailable(AvailabilityState.UNKNOWN, unavailable)


def _unknown_bool(reason: str) -> AvailableBool:
    provenance = FieldProvenance.unavailable(AvailabilityState.UNKNOWN, (reason,))
    return AvailableBool.unavailable(AvailabilityState.UNKNOWN, provenance)


def _first_drop(
    replay: _ReplayState, residual: int | None, provenance: FieldProvenance,
) -> AvailableInt:
    value = None if residual is None else min(residual, OJAMA_MAX_DROP_PER_TURN)
    return _causal_int(replay, value, provenance)


def _leftover(
    replay: _ReplayState, residual: int | None, provenance: FieldProvenance,
) -> AvailableInt:
    value = None if residual is None else max(0, residual - OJAMA_MAX_DROP_PER_TURN)
    return _causal_int(replay, value, provenance)


def _active_chain_side(
    replay: _ReplayState, provenance: FieldProvenance,
) -> AvailableSide:
    active = [side for side in SIDES if replay.chain_active[side]]
    if not active:
        return AvailableSide.known_none(provenance)
    value = active[0] if len(active) == 1 else "both"
    return AvailableSide.known(value, provenance)


def _recipient(replay: _ReplayState, provenance: FieldProvenance) -> AvailableSide:
    balance = replay.causal.observed_balance()
    if balance is None or not _causal_ready(replay):
        unavailable = FieldProvenance.unavailable(
            AvailabilityState.UNSUPPORTED,
            _causal_reason_codes(replay),
        )
        return AvailableSide.unavailable(AvailabilityState.UNSUPPORTED, unavailable)
    if balance == 0:
        return AvailableSide.known_none(provenance)
    return AvailableSide.known("p2" if balance > 0 else "p1", provenance)


def _ledger_provenance(replay: _ReplayState) -> FieldProvenance:
    if replay.last_event is None:
        raise CanonicalObservationAdapterError("ledger provenanceがありません")
    timing = replay.last_event["timing"]
    return FieldProvenance.known(
        tuple(sorted(set(replay.accepted_event_ids))),
        int(timing["available_frame"]),
        int(timing["available_ms"]),
        kind=ProvenanceKind.CAUSAL_LEDGER,
    )


def _event_provenance(
    event: Mapping[str, Any], kind: ProvenanceKind,
) -> FieldProvenance:
    timing = event["timing"]
    return FieldProvenance.known(
        (str(event["event_id"]),),
        int(timing["available_frame"]),
        int(timing["available_ms"]),
        kind=kind,
    )


def _build_quality(replay: _ReplayState) -> ObservationQuality:
    reasons = set(replay.unsupported_reasons)
    reasons.update(replay.causal.reason_codes)
    if replay.pending_disagreement:
        reasons.add("causal_ledger_pending_disagrees_while_settled")
    if not reasons:
        return ObservationQuality(ObservationStatus.READY, ())
    quarantined = not _causal_ready(replay)
    return ObservationQuality(
        ObservationStatus.UNSUPPORTED, tuple(sorted(reasons)), quarantined,
    )


# 学習・本番が分岐した実装を持たないことをAPI上でも固定する。
canonical_observation_for_training = canonical_observation_from_committed_prefix
canonical_observation_for_serving = canonical_observation_from_committed_prefix


__all__ = [
    "CANONICAL_ADAPTER_VERSION", "CanonicalCutoff",
    "CanonicalObservationAdapterError", "canonical_observation_for_serving",
    "canonical_observation_for_training",
    "canonical_observation_from_committed_prefix", "iter_canonical_observations",
]
