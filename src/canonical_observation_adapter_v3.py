"""V2 event adapter出力を確定攻撃分離済みcanonical V3へ変換する。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace

from src.canonical_observation_adapter_v2 import (
    CanonicalCutoff,
    CanonicalObservationAdapterError,
    FuturePolicy,
    canonical_observation_from_committed_prefix as _v2_from_prefix,
    iter_canonical_observations as _iter_v2_observations,
)
from src.canonical_observation_v2 import (
    AvailabilityState,
    AvailableInt,
    AvailableSide,
    CanonicalObservationV2,
    CausalLedgerSideSnapshot,
    CausalLedgerSnapshot,
    FieldProvenance,
    ProvenanceKind,
    validate_canonical_observation as validate_v2_observation,
)
from src.canonical_observation_v3 import (
    CanonicalObservationV3,
    ProjectedDropProvenanceStatus,
    ProjectedDropProvenanceV1,
)
from src.event_source_v1 import CommittedBatch
from src.projected_state_observation_v1 import ProjectedStateObservationV1


CANONICAL_ADAPTER_VERSION = "canonical-observation-adapter/v3"
SIDES = ("p1", "p2")


def canonical_observation_v3_from_v2(
    observation: CanonicalObservationV2,
    projected_state: ProjectedStateObservationV1 | None = None,
) -> CanonicalObservationV3:
    """V2の確定pendingだけを権威値としてV3を純粋変換する。"""
    validate_v2_observation(observation)
    if projected_state is not None and not isinstance(
        projected_state, ProjectedStateObservationV1,
    ):
        raise CanonicalObservationAdapterError("projected stateの型が不正です")
    ledger = _convert_ledger(observation, projected_state)
    projected_provenance = _projected_drop_provenance(ledger, projected_state)
    return CanonicalObservationV3(
        observation_id=observation.observation_id,
        source_video_id=observation.source_video_id,
        build_id=observation.build_id, game_idx=observation.game_idx,
        through_event_seq=observation.through_event_seq,
        cutoff_frame=observation.cutoff_frame, cutoff_ms=observation.cutoff_ms,
        boundary_segment=observation.boundary_segment,
        p1=observation.p1, p2=observation.p2, ledger=ledger,
        provenance=observation.provenance, quality=observation.quality,
        projected_drop_provenance=projected_provenance,
    )


def canonical_observation_from_committed_prefix(
    batches: Iterable[CommittedBatch],
    cutoff: CanonicalCutoff,
    *,
    future_policy: FuturePolicy = "reject",
    projected_state: ProjectedStateObservationV1 | None = None,
) -> CanonicalObservationV3 | None:
    """V2 prefix replayとV3純変換を結ぶ学習・本番共通入口。"""
    observation = _v2_from_prefix(batches, cutoff, future_policy=future_policy)
    if observation is None:
        return None
    return canonical_observation_v3_from_v2(observation, projected_state)


def iter_canonical_observations(
    batches: Iterable[CommittedBatch],
    projected_states: Mapping[int, ProjectedStateObservationV1] | None = None,
) -> Iterator[CanonicalObservationV3]:
    """一回replayの各観測を同じV3純変換へ通す。"""
    by_sequence = {} if projected_states is None else projected_states
    for observation in _iter_v2_observations(batches):
        projected = by_sequence.get(observation.through_event_seq)
        yield canonical_observation_v3_from_v2(observation, projected)


def _convert_ledger(
    observation: CanonicalObservationV2,
    projected: ProjectedStateObservationV1 | None,
) -> CausalLedgerSnapshot:
    source = observation.ledger
    recipient = _finalized_recipient(source)
    sides = {
        side: _convert_side(
            getattr(source, side), side, recipient.value, observation, projected,
        )
        for side in SIDES
    }
    return CausalLedgerSnapshot(
        p1=sides["p1"], p2=sides["p2"],
        active_chain_side=source.active_chain_side,
        recipient=recipient, ledger_prefix_digest=source.ledger_prefix_digest,
        provenance=source.provenance,
    )


def _convert_side(
    source: CausalLedgerSideSnapshot,
    side: str,
    recipient: str | None,
    observation: CanonicalObservationV2,
    projected: ProjectedStateObservationV1 | None,
) -> CausalLedgerSideSnapshot:
    pending = source.pending_garbage
    first, leftover = _drop_breakdown(
        pending, side, recipient, observation, projected,
    )
    return replace(
        source, finalized_unsettled_attack=pending,
        post_cancel_residual=pending, first_drop_amount=first,
        leftover_after_first_drop=leftover,
    )


def _finalized_recipient(ledger: CausalLedgerSnapshot) -> AvailableSide:
    first, second = ledger.p1.pending_garbage, ledger.p2.pending_garbage
    known_states = {AvailabilityState.KNOWN, AvailabilityState.KNOWN_ZERO}
    if first.availability not in known_states or second.availability not in known_states:
        state, reasons = _merged_unavailability(first, second)
        return AvailableSide.unavailable(
            state, FieldProvenance.unavailable(state, reasons),
        )
    recipient = "p1" if int(first.value) else ("p2" if int(second.value) else None)
    provenance = _derived_provenance(ledger.provenance)
    return (AvailableSide.known_none(provenance) if recipient is None
            else AvailableSide.known(recipient, provenance))


def _merged_unavailability(
    first: AvailableInt, second: AvailableInt,
) -> tuple[AvailabilityState, tuple[str, ...]]:
    states = {first.availability, second.availability}
    if AvailabilityState.INTEGRITY_FAULT in states:
        state = AvailabilityState.INTEGRITY_FAULT
    elif AvailabilityState.UNSUPPORTED in states:
        state = AvailabilityState.UNSUPPORTED
    else:
        state = AvailabilityState.UNKNOWN
    reasons = set(first.provenance.reason_codes) | set(second.provenance.reason_codes)
    return state, tuple(sorted(reasons or {"finalized_pending_unavailable"}))


def _drop_breakdown(
    pending: AvailableInt,
    side: str,
    recipient: str | None,
    observation: CanonicalObservationV2,
    projected: ProjectedStateObservationV1 | None,
) -> tuple[AvailableInt, AvailableInt]:
    known_states = {AvailabilityState.KNOWN, AvailabilityState.KNOWN_ZERO}
    if pending.availability not in known_states:
        return pending, pending
    amount = int(pending.value)
    if amount == 0:
        return pending, pending
    reason = _projection_rejection_reason(
        observation, projected, side, recipient, amount,
    )
    if reason is not None:
        missing = FieldProvenance.unavailable(AvailabilityState.UNKNOWN, (reason,))
        unknown = AvailableInt.unavailable(AvailabilityState.UNKNOWN, missing)
        return unknown, unknown
    provenance = _derived_provenance(observation.ledger.provenance)
    return (
        AvailableInt.known(int(projected.first_drop_amount.value), provenance),
        AvailableInt.known(int(projected.leftover_after_first_drop.value), provenance),
    )


def _projection_rejection_reason(
    observation: CanonicalObservationV2,
    projected: ProjectedStateObservationV1 | None,
    side: str,
    recipient: str | None,
    amount: int,
) -> str | None:
    if projected is None:
        return "guaranteed_projection_not_observed"
    if not _same_prefix(observation, projected):
        return "guaranteed_projection_prefix_mismatch"
    if projected.gate_status != "guaranteed":
        return "guaranteed_projection_gate_not_met"
    first, leftover = projected.first_drop_amount, projected.leftover_after_first_drop
    if projected.recipient != recipient or projected.recipient != side:
        return "guaranteed_projection_recipient_mismatch"
    if not first.present or not leftover.present:
        return "guaranteed_projection_amount_missing"
    if int(first.value) + int(leftover.value) != amount:
        return "guaranteed_projection_amount_mismatch"
    return None


def _same_prefix(
    observation: CanonicalObservationV2,
    projected: ProjectedStateObservationV1,
) -> bool:
    expected = (
        observation.source_video_id, observation.build_id,
        observation.provenance.attempt_id, observation.game_idx,
        observation.boundary_segment, observation.through_event_seq,
        observation.provenance.event_prefix_semantic_digest,
    )
    actual = (
        projected.source_video_id, projected.build_id, projected.attempt_id,
        projected.game_idx, projected.boundary_segment,
        projected.through_event_seq, projected.causal_cutoff_digest,
    )
    return expected == actual


def _projected_drop_provenance(
    ledger: CausalLedgerSnapshot,
    projected: ProjectedStateObservationV1 | None,
) -> ProjectedDropProvenanceV1 | None:
    if projected is None:
        return None
    known_states = {AvailabilityState.KNOWN, AvailabilityState.KNOWN_ZERO}
    recipients = tuple(
        side for side in SIDES
        if int(getattr(ledger, side).pending_garbage.value or 0) > 0
        and all(item.availability in known_states for item in (
            getattr(ledger, side).first_drop_amount,
            getattr(ledger, side).leftover_after_first_drop,
        ))
    )
    if len(recipients) != 1:
        return None
    return ProjectedDropProvenanceV1(
        status=ProjectedDropProvenanceStatus.VERIFIED,
        origin_recipient=recipients[0], current_recipient=recipients[0],
        input_digest=projected.input_digest,
        adapter_hash=projected.adapter_hash, physics_hash=projected.physics_hash,
    )


def _derived_provenance(source: FieldProvenance) -> FieldProvenance:
    return FieldProvenance.known(
        source.source_event_ids, int(source.available_frame), int(source.available_ms),
        kind=ProvenanceKind.RULE_DERIVED,
        confidence_milli=source.confidence_milli,
    )


# 学習・本番に別実装を持たせないことをalias同一性で固定する。
canonical_observation_v3_for_training = canonical_observation_v3_from_v2
canonical_observation_v3_for_serving = canonical_observation_v3_from_v2
canonical_observation_for_training = canonical_observation_from_committed_prefix
canonical_observation_for_serving = canonical_observation_from_committed_prefix


__all__ = [
    "CANONICAL_ADAPTER_VERSION", "CanonicalCutoff",
    "CanonicalObservationAdapterError", "canonical_observation_for_serving",
    "canonical_observation_for_training",
    "canonical_observation_from_committed_prefix",
    "canonical_observation_v3_for_serving",
    "canonical_observation_v3_for_training", "canonical_observation_v3_from_v2",
    "iter_canonical_observations",
]
