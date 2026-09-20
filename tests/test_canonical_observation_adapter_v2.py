"""Canonical観測adapterの因果prefix・境界・共通入口契約。"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from src.canonical_observation_adapter_v2 import (
    CanonicalCutoff,
    CanonicalObservationAdapterError,
    canonical_observation_for_serving,
    canonical_observation_for_training,
    canonical_observation_from_committed_prefix,
    iter_canonical_observations,
)
from src.canonical_observation_v2 import AvailabilityState, ObservationStatus
from src.event_source_v1 import CommittedBatch, SCHEMA_VERSION, semantic_events_sha256


SOURCE = "video-canonical-test"
BUILD = "build-canonical-test"
ATTEMPT = "attempt-canonical-test"
EMPTY_GRID = [[0 for _ in range(6)] for _ in range(13)]


def _event(
    seq: int,
    event_type: str,
    side: str,
    *,
    marker: int = 0,
    include_next: bool = True,
    amount: int = 0,
    relation_state: str = "single_observed_chain",
) -> dict[str, Any]:
    payload, assertion = _payload(
        event_type, marker, include_next, amount, relation_state,
    )
    return {
        "record_kind": "event",
        "schema_version": SCHEMA_VERSION,
        "event_id": f"{BUILD}:{seq}",
        "event_type": event_type,
        "side": side,
        "source_video_id": SOURCE,
        "build_id": BUILD,
        "attempt_id": ATTEMPT,
        "availability_batch_id": "pending",
        "batch_index": 0,
        "batch_size": 1,
        "seq": seq,
        "assertion": assertion,
        "evidence": [],
        "checks": [],
        "missing_information": [],
        "heavy_evidence_refs": [],
        "relations": _relations(seq, event_type),
        "payload": payload,
        "timing": _timing(seq),
    }


def _payload(
    event_type: str, marker: int, include_next: bool,
    amount: int, relation_state: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    assertion = {"state": "confirmed", "value_form": "exact"}
    if event_type == "stable_board_observed":
        grid = copy.deepcopy(EMPTY_GRID)
        grid[12][0] = marker
        context: dict[str, Any] = {
            "all_clear_pending": False, "score": 0, "tsumo_count": 8,
        }
        if include_next:
            context.update({
                "next_pair": {"first": 1, "second": 2},
                "double_next_pair": {"first": 3, "second": 4},
            })
        payload = {
            "grid": grid,
            "unknown_mask": [[0 for _ in range(6)] for _ in range(13)],
            "color_cell_count": int(marker != 0),
            "garbage_cell_count": 0,
            "occupied_cell_count": int(marker != 0),
            "unknown_cell_count": 0,
            "board_provenance": "observed",
            "observed_context": context,
        }
    elif event_type == "attack_provisional_updated":
        assertion = {"state": "provisional", "value_form": "exact"}
        payload = {
            "provisional_generated_amount": amount,
            "provisional_score": amount * 70,
            "chain_count": 2,
            "effective_rate": 70,
        }
    elif event_type == "attack_finalized":
        payload = {
            "generated_amount": amount,
            "effective_rate": 70,
            "chain_relation_state": relation_state,
        }
    elif event_type == "garbage_sent":
        payload = {"recipient": "p2", "sent_amount": amount}
    elif event_type == "match_boundary_evidence":
        payload = {"opening_game_index_unverified": 1}
    else:
        payload = {}
    return payload, assertion


def _relations(seq: int, event_type: str) -> dict[str, Any]:
    relations: dict[str, Any] = {
        "revision": {"action": "none", "target_event_ids": []},
    }
    if event_type in {"attack_provisional_updated", "attack_finalized"}:
        relations["attack_id"] = f"attack-{seq}"
    return relations


def _timing(seq: int) -> dict[str, int]:
    return {
        "occurred_earliest_frame": seq,
        "occurred_earliest_ms": seq * 10,
        "occurred_latest_frame": seq,
        "occurred_latest_ms": seq * 10,
        "available_frame": seq,
        "available_ms": seq * 10,
    }


def _batch(event: dict[str, Any]) -> CommittedBatch:
    normalized = copy.deepcopy(event)
    batch_id = f"batch-{normalized['seq']}"
    normalized["availability_batch_id"] = batch_id
    events = (normalized,)
    return CommittedBatch(batch_id, events, 0, 0, semantic_events_sha256(events))


def _base_batches(*, include_next: bool = True) -> tuple[CommittedBatch, ...]:
    return (
        _batch(_event(0, "stable_board_observed", "p1", marker=1,
                      include_next=include_next)),
        _batch(_event(1, "stable_board_observed", "p2", marker=2)),
    )


def test_same_accepted_prefix_is_causal_despite_different_future() -> None:
    base = _base_batches()
    future_a = _batch(_event(2, "stable_board_observed", "p1", marker=3))
    future_b = _batch(_event(2, "stable_board_observed", "p1", marker=4))
    cutoff = CanonicalCutoff(1, 10)

    first = canonical_observation_from_committed_prefix(
        (*base, future_a), cutoff, future_policy="exclude",
    )
    second = canonical_observation_from_committed_prefix(
        (*base, future_b), cutoff, future_policy="exclude",
    )

    assert first is not None and second is not None
    assert first.canonical_json_bytes() == second.canonical_json_bytes()
    assert first.digest == second.digest


def test_future_event_is_rejected_by_default() -> None:
    future = _batch(_event(2, "stable_board_observed", "p1"))
    with pytest.raises(CanonicalObservationAdapterError, match="cutoff後"):
        canonical_observation_from_committed_prefix(
            (*_base_batches(), future), CanonicalCutoff(1, 10),
        )


def test_training_and_serving_are_the_same_core_and_bit_identical() -> None:
    assert canonical_observation_for_training is canonical_observation_for_serving
    batches = _base_batches()
    cutoff = CanonicalCutoff(1, 10)
    training = canonical_observation_for_training(batches, cutoff)
    serving = canonical_observation_for_serving(copy.deepcopy(batches), cutoff)
    assert training is not None and serving is not None
    assert training.canonical_json_bytes() == serving.canonical_json_bytes()
    assert training.digest == serving.digest


def test_streaming_iterator_matches_prefix_adapter_bit_identically() -> None:
    """大量学習用の一回replayも本番prefix coreと同じDTOを返す。"""

    batches = _base_batches()
    streamed = list(iter_canonical_observations(batches))
    serving = canonical_observation_for_serving(batches, CanonicalCutoff(1, 10))

    assert len(streamed) == 1
    assert serving is not None
    assert streamed[0].canonical_json_bytes() == serving.canonical_json_bytes()


def test_nonstable_event_freezes_last_confirmed_board() -> None:
    chain = _batch(_event(2, "chain_started", "p1"))
    result = canonical_observation_from_committed_prefix(
        (*_base_batches(), chain), CanonicalCutoff(2, 20),
    )
    assert result is not None and result.p1.board.grid is not None
    assert result.p1.board.grid[12][0] == 1
    assert result.ledger.p1.chain_active.value is True
    assert result.ledger.p1.chain_step.availability == AvailabilityState.UNKNOWN


def test_zero_provisional_pending_mismatch_is_unsupported_in_shared_adapter() -> None:
    batches = (
        *_base_batches(),
        _batch(_event(2, "attack_provisional_updated", "p1", amount=0)),
        _batch(_event(3, "stable_board_observed", "p1", marker=3)),
        _batch(_event(4, "garbage_sent", "p1", amount=12)),
    )

    result = canonical_observation_for_training(batches, CanonicalCutoff(4, 40))

    assert result is not None
    assert result.quality.status == ObservationStatus.UNSUPPORTED
    assert result.quality.quarantined is True
    assert "causal_ledger_pending_disagrees_while_settled" in result.quality.reason_codes
    assert result.ledger.p2.pending_garbage.availability == AvailabilityState.UNSUPPORTED


def test_formal_boundary_discards_old_board_and_ledger_state() -> None:
    old = (*_base_batches(), _batch(
        _event(2, "attack_provisional_updated", "p1", amount=12),
    ))
    boundary = _batch(_event(3, "match_boundary_evidence", "system"))
    new_p1 = _batch(_event(4, "stable_board_observed", "p1", marker=3))
    cutoff = CanonicalCutoff(4, 40)
    assert canonical_observation_from_committed_prefix(
        (*old, boundary, new_p1), cutoff,
    ) is None

    new_p2 = _batch(_event(5, "stable_board_observed", "p2", marker=4))
    result = canonical_observation_from_committed_prefix(
        (*old, boundary, new_p1, new_p2), CanonicalCutoff(5, 50),
    )
    assert result is not None
    assert result.boundary_segment == result.game_idx == 1
    assert result.p1.board.grid is not None and result.p1.board.grid[12][0] == 3
    assert result.ledger.p2.post_cancel_residual.value == 0
    assert result.ledger.p1.provisional_generated.value == 0
    assert result.ledger.active_chain_side.availability == AvailabilityState.KNOWN_ZERO


def test_posthoc_future_is_excluded_but_accepted_posthoc_is_rejected() -> None:
    winner = _batch(_event(2, "winner_observed", "system"))
    excluded = canonical_observation_from_committed_prefix(
        (*_base_batches(), winner), CanonicalCutoff(1, 10), future_policy="exclude",
    )
    assert excluded is not None
    with pytest.raises(CanonicalObservationAdapterError, match="posthoc"):
        canonical_observation_from_committed_prefix(
            (*_base_batches(), winner), CanonicalCutoff(2, 20),
        )


@pytest.mark.parametrize(
    ("assertion", "provenance"),
    [
        ({"state": "provisional", "value_form": "exact"}, "observed"),
        ({"state": "confirmed", "value_form": "exact"}, "physics_projected"),
    ],
)
def test_only_stable_confirmed_observed_board_is_accepted(
    assertion: dict[str, str], provenance: str,
) -> None:
    broken = _event(0, "stable_board_observed", "p1")
    broken["assertion"] = assertion
    broken["payload"]["board_provenance"] = provenance
    batches = (_batch(broken), _batch(_event(1, "stable_board_observed", "p2")))
    with pytest.raises(CanonicalObservationAdapterError, match="STABLE"):
        canonical_observation_from_committed_prefix(batches, CanonicalCutoff(1, 10))


def test_missing_piece_is_unknown_and_known_zero_is_not_conflated() -> None:
    result = canonical_observation_from_committed_prefix(
        _base_batches(include_next=False), CanonicalCutoff(1, 10),
    )
    assert result is not None and result.quality.status == ObservationStatus.READY
    assert result.p1.pieces.next.axis.availability == AvailabilityState.UNKNOWN
    assert result.p1.pieces.next.axis.value is None
    assert result.ledger.p1.pending_garbage.availability == AvailabilityState.KNOWN_ZERO
    assert result.ledger.p1.pending_garbage.value == 0


def test_partial_piece_pair_preserves_known_and_unknown_components() -> None:
    p1 = _event(0, "stable_board_observed", "p1")
    p1["payload"]["observed_context"]["next_pair"] = {"first": 4}
    batches = (_batch(p1), _batch(_event(1, "stable_board_observed", "p2")))
    result = canonical_observation_from_committed_prefix(
        batches, CanonicalCutoff(1, 10),
    )
    assert result is not None
    assert result.p1.pieces.next.axis.value == 4
    assert result.p1.pieces.next.axis.availability == AvailabilityState.KNOWN
    assert result.p1.pieces.next.child.value is None
    assert result.p1.pieces.next.child.availability == AvailabilityState.UNKNOWN


def test_ambiguous_causal_ledger_is_unsupported_not_zero() -> None:
    ambiguous = _batch(_event(
        2, "attack_finalized", "p1", amount=7, relation_state="ambiguous",
    ))
    result = canonical_observation_from_committed_prefix(
        (*_base_batches(), ambiguous), CanonicalCutoff(2, 20),
    )
    assert result is not None and result.quality.status == ObservationStatus.UNSUPPORTED
    pending = result.ledger.p2.pending_garbage
    assert pending.availability == AvailabilityState.UNSUPPORTED
    assert pending.value is None


def test_tampered_committed_batch_is_rejected() -> None:
    batch = _batch(_event(0, "stable_board_observed", "p1"))
    batch.events[0]["payload"]["grid"][12][0] = 4
    with pytest.raises(CanonicalObservationAdapterError, match="semantic hash"):
        canonical_observation_from_committed_prefix((batch,), CanonicalCutoff(0, 0))
