"""CanonicalObservationV2の欠測・serialization・左右交換・拒否契約。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.board import COLOR_UNKNOWN
from src.canonical_observation_v2 import (
    AvailabilityState,
    AvailableBool,
    AvailableColor,
    AvailableInt,
    AvailableSide,
    CANONICAL_OBSERVATION_SCHEMA_VERSION,
    CanonicalObservationError,
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


SHA = "a" * 64


def _known_provenance(event: str = "build:1") -> FieldProvenance:
    return FieldProvenance.known((event,), 100, 1000, confidence_milli=995)


def _missing(state: AvailabilityState, reason: str) -> FieldProvenance:
    return FieldProvenance.unavailable(state, (reason,))


def _board(marker: int, event: str) -> StableBoardObservation:
    rows = [[0 for _ in range(6)] for _ in range(13)]
    rows[12][0] = marker
    rows[0][5] = COLOR_UNKNOWN
    grid = tuple(tuple(row) for row in rows)
    mask = tuple(tuple(cell != COLOR_UNKNOWN for cell in row) for row in grid)
    return StableBoardObservation.known(grid, mask, _known_provenance(event))


def _pair(first: int, second: int, event: str) -> PiecePairObservation:
    provenance = _known_provenance(event)
    return PiecePairObservation(
        AvailableColor.known(first, provenance), AvailableColor.known(second, provenance),
    )


def _unknown_pair(reason: str) -> PiecePairObservation:
    provenance = _missing(AvailabilityState.UNKNOWN, reason)
    return PiecePairObservation(
        AvailableColor.unavailable(AvailabilityState.UNKNOWN, provenance),
        AvailableColor.unavailable(AvailabilityState.UNKNOWN, provenance),
    )


def _side(marker: int, prefix: str) -> CanonicalSideObservation:
    known = _known_provenance(f"{prefix}:state")
    return CanonicalSideObservation(
        board=_board(marker, f"{prefix}:board"),
        pieces=PieceQueueObservation(
            _unknown_pair("current_piece_not_recorded"),
            _pair(1, 2, f"{prefix}:next"), _pair(3, 4, f"{prefix}:dnext"),
        ),
        all_clear_pending=AvailableBool.known(False, known),
        raw_score=AvailableInt.known(0, known), tsumo_count=AvailableInt.known(12, known),
    )


def _ledger_side(prefix: str, pending: int, active: bool) -> CausalLedgerSideSnapshot:
    provenance = FieldProvenance.known(
        (f"{prefix}:ledger",), 100, 1000, kind=ProvenanceKind.CAUSAL_LEDGER,
    )
    quantity = lambda value: AvailableInt.known(value, provenance)
    flag = lambda value: AvailableBool.known(value, provenance)
    return CausalLedgerSideSnapshot(
        pending_garbage=quantity(pending), effective_rate=quantity(70),
        chain_active=flag(active), chain_step=quantity(2 if active else 0),
        provisional_generated=quantity(4 if active else 0),
        provisional_score=quantity(280 if active else 0),
        provisional_chain_count=quantity(2 if active else 0),
        finalized_unsettled_attack=quantity(pending), post_cancel_residual=quantity(pending),
        send_waiting=flag(active), placement_waiting=flag(False),
        chain_end_waiting=flag(active), garbage_falling=flag(False),
        first_drop_amount=quantity(min(pending, 30)),
        leftover_after_first_drop=quantity(max(0, pending - 30)),
    )


def _observation() -> CanonicalObservationV2:
    ledger_provenance = FieldProvenance.known(
        ("ledger:snapshot",), 100, 1000, kind=ProvenanceKind.CAUSAL_LEDGER,
    )
    side_provenance = _known_provenance("ledger:direction")
    ledger = CausalLedgerSnapshot(
        p1=_ledger_side("p1", 0, True), p2=_ledger_side("p2", 35, False),
        active_chain_side=AvailableSide.known("p1", side_provenance),
        recipient=AvailableSide.known("p2", side_provenance),
        ledger_prefix_digest=SHA, provenance=ledger_provenance,
    )
    return CanonicalObservationV2(
        observation_id="obs-1", source_video_id="video-a", build_id="build-a",
        game_idx=3, through_event_seq=42, cutoff_frame=100, cutoff_ms=1000,
        boundary_segment=3, p1=_side(1, "p1"), p2=_side(2, "p2"), ledger=ledger,
        provenance=ObservationProvenance(SHA, source_group_id="youtube-a"),
        quality=ObservationQuality(ObservationStatus.READY, ()),
    )


def test_known_zero_unknown_unsupported_and_fault_are_distinct() -> None:
    known = _known_provenance()
    zero = AvailableInt.known(0, known)
    unknown = AvailableInt.unavailable(
        AvailabilityState.UNKNOWN, _missing(AvailabilityState.UNKNOWN, "not_observed"),
    )
    unsupported = AvailableInt.unavailable(
        AvailabilityState.UNSUPPORTED,
        _missing(AvailabilityState.UNSUPPORTED, "source_schema_unsupported"),
    )
    fault = AvailableInt.unavailable(
        AvailabilityState.INTEGRITY_FAULT,
        _missing(AvailabilityState.INTEGRITY_FAULT, "ledger_mismatch"),
    )
    assert [item.availability for item in (zero, unknown, unsupported, fault)] == [
        AvailabilityState.KNOWN_ZERO, AvailabilityState.UNKNOWN,
        AvailabilityState.UNSUPPORTED, AvailabilityState.INTEGRITY_FAULT,
    ]
    assert zero.value == 0
    assert unknown.value is unsupported.value is fault.value is None


def test_pending_and_exchange_residual_remain_distinct_raw_quantities() -> None:
    observation = _observation()
    provenance = _known_provenance("p2:pending-meter")
    changed = replace(
        observation.ledger.p2,
        pending_garbage=AvailableInt.known(47, provenance),
    )
    assert changed.pending_garbage.value == 47
    assert changed.post_cancel_residual.value == 35


def test_canonical_serialization_roundtrip_and_digest_are_stable() -> None:
    observation = _observation()
    payload = observation.canonical_json_bytes()
    restored = CanonicalObservationV2.from_json_bytes(payload)
    assert restored == observation
    assert CanonicalObservationV2.from_dict(observation.to_dict()) == observation
    assert restored.to_dict() == observation.to_dict()
    assert isinstance(observation.to_dict()["p1"]["board"]["grid"], list)
    assert restored.canonical_json_bytes() == payload
    assert restored.digest == observation.digest == observation.input_digest
    assert observation.to_dict()["schema_version"] == CANONICAL_OBSERVATION_SCHEMA_VERSION


def test_side_swap_is_complete_and_twice_bit_identical() -> None:
    observation = _observation()
    swapped = observation.swap_sides()
    assert swapped.p1 == observation.p2
    assert swapped.p2 == observation.p1
    assert swapped.ledger.p1 == observation.ledger.p2
    assert swapped.ledger.active_chain_side.value == "p2"
    assert swapped.ledger.recipient.value == "p1"
    twice = swapped.swap_sides()
    assert twice == observation
    assert twice.canonical_json_bytes() == observation.canonical_json_bytes()
    assert twice.digest == observation.digest


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (lambda: AvailableInt(0, AvailabilityState.KNOWN, _known_provenance()), "known_zero"),
        (lambda: AvailableInt(None, AvailabilityState.KNOWN_ZERO, _known_provenance()), "値がありません"),
        (lambda: AvailableColor.known(9, _known_provenance()), "1..5"),
        (lambda: AvailableBool(False, AvailabilityState.UNKNOWN,
                               _missing(AvailabilityState.UNKNOWN, "missing")), "value=None"),
        (lambda: AvailableSide(None, "known_zero", _known_provenance()), "availability"),
    ],
)
def test_invalid_available_values_are_rejected(value: object, match: str) -> None:
    with pytest.raises(CanonicalObservationError, match=match):
        value()


def test_invalid_board_mask_and_future_provenance_are_rejected() -> None:
    board = _board(1, "board")
    bad_mask = tuple(tuple(True for _ in range(6)) for _ in range(13))
    with pytest.raises(CanonicalObservationError, match="known mask"):
        StableBoardObservation.known(board.grid, bad_mask, board.provenance)
    observation = _observation()
    future = FieldProvenance.known(("future",), 101, 1001)
    bad_side = replace(observation.p1, raw_score=AvailableInt.known(10, future))
    with pytest.raises(CanonicalObservationError, match="未来"):
        replace(observation, p1=bad_side)


def test_provenance_and_unsupported_quality_are_strict() -> None:
    with pytest.raises(CanonicalObservationError, match="event/timing"):
        FieldProvenance(ProvenanceKind.DIRECT_OBSERVATION, (), None, None)
    with pytest.raises(CanonicalObservationError, match="理由"):
        FieldProvenance(ProvenanceKind.UNKNOWN, (), None, None)
    observation = _observation()
    unavailable = AvailableColor.unavailable(
        AvailabilityState.UNSUPPORTED,
        _missing(AvailabilityState.UNSUPPORTED, "piece_schema_unsupported"),
    )
    pair = replace(observation.p1.pieces.current, axis=unavailable)
    pieces = replace(observation.p1.pieces, current=pair)
    with pytest.raises(CanonicalObservationError, match="unsupported"):
        replace(observation, p1=replace(observation.p1, pieces=pieces))


def test_schema_extra_key_and_ledger_inconsistency_are_rejected() -> None:
    observation = _observation()
    payload = observation.to_dict()
    payload["unexpected"] = True
    with pytest.raises(CanonicalObservationError, match="key集合"):
        CanonicalObservationV2.from_dict(payload)
    with pytest.raises(CanonicalObservationError, match="recipient"):
        replace(
            observation.ledger,
            recipient=AvailableSide.known("p1", _known_provenance("wrong-recipient")),
        )


def test_integrity_fault_requires_hard_fault_quality() -> None:
    observation = _observation()
    provenance = _missing(AvailabilityState.INTEGRITY_FAULT, "board_hash_mismatch")
    board = StableBoardObservation.unavailable(AvailabilityState.INTEGRITY_FAULT, provenance)
    with pytest.raises(CanonicalObservationError, match="quality"):
        replace(observation, p1=replace(observation.p1, board=board))
    fault_quality = ObservationQuality(
        ObservationStatus.INTEGRITY_FAULT, ("board_hash_mismatch",), quarantined=True,
    )
    fault = replace(observation, p1=replace(observation.p1, board=board), quality=fault_quality)
    assert fault.quality.status == ObservationStatus.INTEGRITY_FAULT
