"""確定送付と物理着地を利用可能時刻のまま交差検証する。"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence


HIGH_CONFIDENCE_EVIDENCE = frozenset({
    "raw_exact", "raw_occupancy_corroborated",
})


@dataclass(frozen=True, slots=True)
class SentSupplyEvidenceV1:
    """相殺後に相手へ送られた確定量。"""

    event_id: str
    sequence: int
    game_idx: int
    recipient: str
    amount: int
    occurred_earliest_frame: int
    occurred_latest_frame: int
    available_frame: int
    causal_candidate_start_frame: int | None
    causal_candidate_start_sequence: int | None
    chain_relation_state: str


@dataclass(frozen=True, slots=True)
class PhysicalCandidateEvidenceV1:
    """盤面差から得た正方向のおじゃま候補。"""

    event_id: str
    sequence: int
    game_idx: int
    recipient: str
    amount: int
    evidence: str
    occurred_earliest_frame: int
    occurred_latest_frame: int
    available_frame: int


@dataclass(frozen=True, slots=True)
class SupplyAllocationV1:
    """一つの盤面候補へ割り当てた確定送付量。"""

    sent_event_id: str
    sent_sequence: int
    amount: int
    sent_available_frame: int
    timing_state: str
    occurrence_overlap: bool
    chain_relation_state: str
    confirmed_balance_committed: bool


@dataclass(frozen=True, slots=True)
class ExchangeCrossMatchV1:
    """物理候補を観測時点と事後確認に分けた照合結果。"""

    physical_event_id: str
    physical_sequence: int
    game_idx: int
    recipient: str
    visible_amount: int
    evidence: str
    physical_occurred_earliest_frame: int
    physical_occurred_latest_frame: int
    physical_available_frame: int
    as_of_matched_amount: int
    retrospectively_matched_amount: int
    unsupported_amount: int
    as_of_status: str
    retrospective_status: str
    balance_label_state: str
    confirmation_available_frame: int | None
    allocations: tuple[SupplyAllocationV1, ...]


@dataclass(frozen=True, slots=True)
class BoundaryEvidenceV1:
    """正式境界の通番と局所試合番号。"""

    sequence: int
    frame: int
    closing_game_idx: int
    opening_game_idx: int


@dataclass(slots=True)
class CandidateAllocationStateV1:
    """観測時点の割当を固定した後、事後確認だけを追記する状態。"""

    candidate: PhysicalCandidateEvidenceV1
    as_of_matched_amount: int
    remaining_amount: int
    allocations: list[SupplyAllocationV1]


@dataclass(frozen=True, slots=True)
class ExchangeCrossValidationReportV1:
    """未来情報を過去へ戻さない物理・確定会計の交差検証。"""

    event_count: int
    sent_supply_count: int
    sent_supply_amount: int
    physical_candidate_count: int
    physical_candidate_amount: int
    physical_evidence_counts: dict[str, int]
    high_confidence_candidate_count: int
    high_confidence_as_of_supported_count: int
    high_confidence_late_supported_count: int
    high_confidence_late_single_count: int
    high_confidence_late_ambiguous_count: int
    high_confidence_occurrence_overlap_count: int
    high_confidence_unsupported_count: int
    high_confidence_balance_usable_count: int
    high_confidence_balance_quarantined_count: int
    low_confidence_unsupported_count: int
    provisional_used_in_confirmed_balance_count: int
    allocated_supply_amount: int
    allocated_supply_overuse_count: int
    max_late_confirmation_frames: int
    physical_gate_pass: bool
    confirmed_balance_gate_pass: bool
    matches: tuple[ExchangeCrossMatchV1, ...]

    def to_json_value(self) -> dict[str, Any]:
        """監査成果物として保存できるJSON互換値を返す。"""
        value = asdict(self)
        value["schema_version"] = "event-exchange-cross-validation/v1"
        return value


def cross_validate_exchange_events_v1(
    events: Sequence[dict[str, Any]],
) -> ExchangeCrossValidationReportV1:
    """統合出来事列から確定送付と物理盤面差を時刻別に照合する。"""
    boundaries = _boundaries(events)
    by_id = {str(event["event_id"]): event for event in events}
    supplies = _sent_supplies(events, boundaries, by_id)
    candidates = _physical_candidates(events)
    _validate_candidate_game_indices(candidates, boundaries)
    matches = _match_candidates(candidates, supplies)
    evidence = Counter(item.evidence for item in candidates)
    high_matches = [item for item in matches if _is_high(item.evidence)]
    high_as_of = sum(item.as_of_status == "supported" for item in high_matches)
    high_late = sum(item.retrospective_status == "late_supported" for item in high_matches)
    late_single = sum(_late_relation_state(item) == "single" for item in high_matches)
    late_ambiguous = sum(_late_relation_state(item) == "ambiguous" for item in high_matches)
    overlap_count = sum(
        item.unsupported_amount == 0
        and bool(item.allocations)
        and all(allocation.occurrence_overlap for allocation in item.allocations)
        for item in high_matches
    )
    high_unsupported = sum(item.unsupported_amount > 0 for item in high_matches)
    usable_states = {"available_as_of", "retrospective_single_chain"}
    balance_usable = sum(item.balance_label_state in usable_states for item in high_matches)
    balance_quarantined = len(high_matches) - balance_usable
    low_unsupported = sum(
        item.unsupported_amount > 0 and not _is_high(item.evidence)
        for item in matches
    )
    allocated_amount, overuse_count = _allocation_usage(matches, supplies)
    lags = [
        item.confirmation_available_frame
        - _candidate_by_id(candidates, item.physical_event_id).available_frame
        for item in matches
        if item.confirmation_available_frame is not None
        and item.retrospective_status == "late_supported"
    ]
    return ExchangeCrossValidationReportV1(
        len(events), len(supplies), sum(item.amount for item in supplies),
        len(candidates), sum(item.amount for item in candidates),
        dict(sorted(evidence.items())), len(high_matches), high_as_of, high_late,
        late_single, late_ambiguous, overlap_count,
        high_unsupported, balance_usable, balance_quarantined,
        low_unsupported, 0, allocated_amount, overuse_count,
        max(lags, default=0),
        high_unsupported == 0 and overuse_count == 0,
        balance_quarantined == 0 and overuse_count == 0, matches,
    )


def unsupported_high_confidence_game_indices_v1(
    report: ExchangeCrossValidationReportV1,
) -> frozenset[int]:
    """説明不能な高信頼着地を含む局所試合番号を返す。"""
    return frozenset(
        match.game_idx for match in report.matches
        if _is_high(match.evidence) and match.unsupported_amount > 0
    )


def unsupported_high_confidence_quarantine_starts_v1(
    report: ExchangeCrossValidationReportV1,
) -> dict[int, int]:
    """説明不能着地を当時初めて観測した通番を試合ごとに返す。"""

    starts: dict[int, int] = {}
    for match in report.matches:
        if not _is_high(match.evidence) or match.unsupported_amount <= 0:
            continue
        current = starts.get(match.game_idx)
        starts[match.game_idx] = (
            match.physical_sequence if current is None
            else min(current, match.physical_sequence)
        )
    return dict(sorted(starts.items()))


def online_uncertain_quarantine_starts_v1(
    report: ExchangeCrossValidationReportV1,
) -> dict[int, int]:
    """観測時点で未説明だった高信頼着地の最初の通番を返す。

    後から送付確定が届いて説明できても、当時の表示入力を過去へ遡って
    利用可能にしない。学習専用の試合全体隔離とは独立したオンライン規約。
    """

    starts: dict[int, int] = {}
    for match in report.matches:
        if not _is_high(match.evidence) or match.as_of_status == "supported":
            continue
        current = starts.get(match.game_idx)
        starts[match.game_idx] = (
            match.physical_sequence if current is None
            else min(current, match.physical_sequence)
        )
    return dict(sorted(starts.items()))


def audit_exchange_cross_quarantine_v1(
    report: ExchangeCrossValidationReportV1,
) -> dict[str, Any]:
    """隔離対象が確定残高へ混入していないことを独立に検査する。"""
    usable_states = {"available_as_of", "retrospective_single_chain"}
    high = [match for match in report.matches if _is_high(match.evidence)]
    low = [match for match in report.matches if not _is_high(match.evidence)]
    unsafe_committed = sum(
        allocation.confirmed_balance_committed
        for match in high if match.balance_label_state not in usable_states
        for allocation in match.allocations
    )
    usable_uncommitted = sum(
        not allocation.confirmed_balance_committed
        for match in high if match.balance_label_state in usable_states
        for allocation in match.allocations
    )
    low_allocated = sum(len(match.allocations) for match in low)
    unsupported_games = unsupported_high_confidence_game_indices_v1(report)
    count_consistent = (
        len(high) == report.high_confidence_candidate_count
        and len(low) == report.low_confidence_unsupported_count
        and len(high) == report.high_confidence_balance_usable_count
        + report.high_confidence_balance_quarantined_count
    )
    contract_pass = (
        count_consistent and unsafe_committed == 0 and usable_uncommitted == 0
        and low_allocated == 0 and report.allocated_supply_overuse_count == 0
        and report.provisional_used_in_confirmed_balance_count == 0
    )
    return {
        "contract_pass": contract_pass,
        "all_physical_candidates_explained": report.physical_gate_pass,
        "unsupported_high_confidence_count": report.high_confidence_unsupported_count,
        "unsupported_game_count": len(unsupported_games),
        "unsupported_game_indices": sorted(unsupported_games),
        "unsafe_committed_allocation_count": unsafe_committed,
        "usable_uncommitted_allocation_count": usable_uncommitted,
        "low_confidence_allocation_count": low_allocated,
        "count_consistent": count_consistent,
    }


def _sent_supplies(
    events: Sequence[dict[str, Any]], boundaries: Sequence[BoundaryEvidenceV1],
    by_id: Mapping[str, dict[str, Any]],
) -> tuple[SentSupplyEvidenceV1, ...]:
    attacks = _attacks_by_id(events)
    result: list[SentSupplyEvidenceV1] = []
    for event in events:
        if event["event_type"] != "garbage_sent":
            continue
        timing, payload, relations = event["timing"], event["payload"], event["relations"]
        attack = attacks.get(str(relations.get("attack_id", "")))
        causal_start = _causal_start_position(attack, by_id)
        frame = int(timing["available_frame"])
        sequence = int(event["seq"])
        result.append(SentSupplyEvidenceV1(
            str(event["event_id"]), sequence,
            _game_index_at_sequence(boundaries, sequence),
            str(payload["recipient"]), int(payload["sent_amount"]),
            int(timing["occurred_earliest_frame"]),
            int(timing["occurred_latest_frame"]), frame,
            causal_start[0] if causal_start is not None else None,
            causal_start[1] if causal_start is not None else None,
            _chain_relation_state(attack),
        ))
    return tuple(result)


def _physical_candidates(
    events: Sequence[dict[str, Any]],
) -> tuple[PhysicalCandidateEvidenceV1, ...]:
    result: list[PhysicalCandidateEvidenceV1] = []
    for event in events:
        if event["event_type"] != "garbage_landing_board_compared":
            continue
        candidate = _physical_candidate(event)
        if candidate is not None:
            result.append(candidate)
    return tuple(result)


def _physical_candidate(
    event: dict[str, Any],
) -> PhysicalCandidateEvidenceV1 | None:
    payload, timing = event["payload"], event["timing"]
    differences = payload.get("differences")
    if payload.get("before_is_immediate_previous_observation") is not True:
        return None
    if not isinstance(differences, Mapping):
        return None
    amount = differences.get("garbage")
    if not _is_positive_integer(amount) or not _is_garbage_only(differences, amount):
        return None
    if payload.get("confirmed_raw_agreement") is True:
        evidence = "raw_exact"
    elif _occupancy_corroborates(payload, amount):
        evidence = "raw_occupancy_corroborated"
    else:
        evidence = "confirmed_board_only"
    return PhysicalCandidateEvidenceV1(
        str(event["event_id"]), int(event["seq"]),
        int(payload["local_game_index_unverified"]),
        str(event["side"]), amount, evidence,
        int(timing["occurred_earliest_frame"]),
        int(timing["occurred_latest_frame"]), int(timing["available_frame"]),
    )


def _match_candidates(
    candidates: Sequence[PhysicalCandidateEvidenceV1],
    supplies: Sequence[SentSupplyEvidenceV1],
) -> tuple[ExchangeCrossMatchV1, ...]:
    grouped_candidates = _group_by_game_side(candidates)
    grouped_supplies = _group_by_game_side(supplies)
    matches: list[ExchangeCrossMatchV1] = []
    for key in sorted(grouped_candidates):
        supplies_for_key = grouped_supplies[key]
        capacities = {item.event_id: item.amount for item in supplies_for_key}
        states = _allocate_as_of_states(
            grouped_candidates[key], supplies_for_key, capacities,
        )
        ordered = sorted(states, key=lambda item: _retrospective_priority(
            item, supplies_for_key, capacities,
        ))
        for state in ordered:
            _allocate_retrospective(state, supplies_for_key, capacities)
        matches.extend(_finalize_match(state) for state in states)
    return tuple(sorted(matches, key=lambda item: item.physical_sequence))


def _allocate_as_of_states(
    candidates: Sequence[PhysicalCandidateEvidenceV1],
    supplies: Sequence[SentSupplyEvidenceV1], capacities: dict[str, int],
) -> list[CandidateAllocationStateV1]:
    """高信頼候補の観測時点割当を物理通番順に固定する。"""
    states: list[CandidateAllocationStateV1] = []
    for candidate in sorted(candidates, key=lambda item: item.sequence):
        allocations: list[SupplyAllocationV1] = []
        remaining = candidate.amount
        if _is_high(candidate.evidence):
            remaining = _allocate_supply(
                candidate, supplies, capacities, remaining, allocations,
                allow_future=False,
            )
        states.append(CandidateAllocationStateV1(
            candidate, candidate.amount - remaining, remaining, allocations,
        ))
    return states


def _allocate_retrospective(
    state: CandidateAllocationStateV1,
    supplies: Sequence[SentSupplyEvidenceV1], capacities: dict[str, int],
) -> None:
    """観測時点割当を変えず、残容量から事後確認だけを追記する。"""
    if not _is_high(state.candidate.evidence) or state.remaining_amount <= 0:
        return
    state.remaining_amount = _allocate_supply(
        state.candidate, supplies, capacities, state.remaining_amount,
        state.allocations, allow_future=True,
    )


def _finalize_match(state: CandidateAllocationStateV1) -> ExchangeCrossMatchV1:
    """固定済みの観測時点値と事後追記を不変結果へ変換する。"""
    candidate = state.candidate
    as_of, remaining = state.as_of_matched_amount, state.remaining_amount
    retrospective = candidate.amount - as_of - remaining
    label_state = _balance_label_state(
        as_of, candidate.amount, remaining, state.allocations,
    )
    committed = label_state in {
        "available_as_of", "retrospective_single_chain",
    }
    finalized_allocations = tuple(
        replace(item, confirmed_balance_committed=committed)
        for item in state.allocations
    )
    confirmation = max(
        (item.sent_available_frame for item in state.allocations), default=None,
    )
    return ExchangeCrossMatchV1(
        candidate.event_id, candidate.sequence, candidate.game_idx,
        candidate.recipient, candidate.amount, candidate.evidence,
        candidate.occurred_earliest_frame, candidate.occurred_latest_frame,
        candidate.available_frame, as_of, retrospective, remaining,
        "supported" if as_of == candidate.amount else "unknown_at_observation",
        _retrospective_status(retrospective, remaining),
        label_state, confirmation, finalized_allocations,
    )


def _allocate_supply(
    candidate: PhysicalCandidateEvidenceV1,
    supplies: Sequence[SentSupplyEvidenceV1], capacities: dict[str, int],
    remaining: int, allocations: list[SupplyAllocationV1], *, allow_future: bool,
) -> int:
    while remaining > 0:
        available = [
            item for item in supplies
            if capacities[item.event_id] > 0
            and _eligible_supply(candidate, item, allow_future)
        ]
        if not available:
            break
        selected = min(available, key=lambda item: _supply_distance(candidate, item))
        amount = min(remaining, capacities[selected.event_id])
        capacities[selected.event_id] -= amount
        remaining -= amount
        allocations.append(SupplyAllocationV1(
            selected.event_id, selected.sequence, amount, selected.available_frame,
            "future_confirmation" if allow_future else "available_as_of",
            _occurrence_overlaps(candidate, selected),
            selected.chain_relation_state,
            False,
        ))
    return remaining


def _eligible_supply(
    candidate: PhysicalCandidateEvidenceV1,
    supply: SentSupplyEvidenceV1, allow_future: bool,
) -> bool:
    candidate_position = (candidate.available_frame, candidate.sequence)
    supply_position = (supply.available_frame, supply.sequence)
    if allow_future:
        causal_position = _causal_position(supply)
        return (
            supply_position > candidate_position
            and causal_position is not None
            and causal_position < candidate_position
        )
    return supply_position < candidate_position


def _causal_position(supply: SentSupplyEvidenceV1) -> tuple[int, int] | None:
    if supply.causal_candidate_start_frame is None:
        return None
    if supply.causal_candidate_start_sequence is None:
        return None
    return supply.causal_candidate_start_frame, supply.causal_candidate_start_sequence


def _retrospective_priority(
    state: CandidateAllocationStateV1,
    supplies: Sequence[SentSupplyEvidenceV1], capacities: Mapping[str, int],
) -> tuple[int, int, int, int]:
    candidate = state.candidate
    label_state = _preview_retrospective_label(state, supplies, capacities)
    ranks = {
        "available_as_of": 0,
        "retrospective_single_chain": 1,
        "direction_only_timing_mismatch": 2,
        "direction_only_ambiguous_chain": 2,
        "quarantined_unsupported": 3,
    }
    return (
        0 if _is_high(candidate.evidence) else 1,
        ranks[label_state], candidate.available_frame, candidate.sequence,
    )


def _preview_retrospective_label(
    state: CandidateAllocationStateV1, supplies: Sequence[SentSupplyEvidenceV1],
    capacities: Mapping[str, int],
) -> str:
    candidate = state.candidate
    preview_capacities = dict(capacities)
    allocations = list(state.allocations)
    remaining = _allocate_supply(
        candidate, supplies, preview_capacities, state.remaining_amount, allocations,
        allow_future=True,
    )
    return _balance_label_state(
        state.as_of_matched_amount, candidate.amount, remaining, allocations,
    )


def _supply_distance(
    candidate: PhysicalCandidateEvidenceV1, supply: SentSupplyEvidenceV1,
) -> tuple[int, int, int, int]:
    return (
        0 if _occurrence_overlaps(candidate, supply) else 1,
        0 if supply.chain_relation_state == "single_observed_chain" else 1,
        abs(supply.available_frame - candidate.available_frame),
        abs(supply.sequence - candidate.sequence),
    )


def _retrospective_status(retrospective: int, unsupported: int) -> str:
    if unsupported > 0:
        return "unsupported"
    if retrospective > 0:
        return "late_supported"
    return "not_needed"


def _balance_label_state(
    as_of: int, visible: int, unsupported: int,
    allocations: Sequence[SupplyAllocationV1],
) -> str:
    """物理量を確定残高との対応ラベルに使える範囲を返す。"""
    if unsupported > 0:
        return "quarantined_unsupported"
    if as_of == visible:
        return "available_as_of"
    future = [item for item in allocations if item.timing_state == "future_confirmation"]
    single = all(
        item.chain_relation_state == "single_observed_chain" for item in future
    )
    if single and all(item.occurrence_overlap for item in future):
        return "retrospective_single_chain"
    if single:
        return "direction_only_timing_mismatch"
    return "direction_only_ambiguous_chain"


def _attacks_by_id(
    events: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["event_type"] != "attack_finalized":
            continue
        attack_id = str(event["relations"].get("attack_id", ""))
        if attack_id:
            result[attack_id] = event
    return result


def _chain_relation_state(attack: dict[str, Any] | None) -> str:
    if attack is None:
        return "missing_attack_relation"
    value = attack["payload"].get("chain_relation_state")
    return value if isinstance(value, str) else "unknown_chain_relation"


def _late_relation_state(match: ExchangeCrossMatchV1) -> str:
    if match.retrospective_status != "late_supported":
        return "none"
    future = [
        item for item in match.allocations
        if item.timing_state == "future_confirmation"
    ]
    if not future:
        return "none"
    if all(item.chain_relation_state == "single_observed_chain" for item in future):
        return "single"
    return "ambiguous"


def _causal_start_position(
    attack: dict[str, Any] | None, by_id: Mapping[str, dict[str, Any]],
) -> tuple[int, int] | None:
    if attack is None:
        return None
    revision = attack["relations"].get("revision", {})
    targets = revision.get("target_event_ids", []) if isinstance(revision, Mapping) else []
    positions = [
        (
            int(by_id[target]["timing"]["available_frame"]),
            int(by_id[target]["seq"]),
        )
        for target in targets if target in by_id
    ]
    return min(positions, default=None)


def _boundaries(
    events: Sequence[dict[str, Any]],
) -> tuple[BoundaryEvidenceV1, ...]:
    result = tuple(sorted(
        (_boundary_from_event(event) for event in events
         if event["event_type"] == "match_boundary_evidence"),
        key=lambda item: item.sequence,
    ))
    for expected, item in enumerate(result):
        if item.closing_game_idx != expected or item.opening_game_idx != expected + 1:
            raise ValueError("正式境界の局所試合番号が連続していません")
    if len({item.sequence for item in result}) != len(result):
        raise ValueError("正式境界の通番が重複しています")
    return result


def _boundary_from_event(event: dict[str, Any]) -> BoundaryEvidenceV1:
    payload = event["payload"]
    return BoundaryEvidenceV1(
        int(event["seq"]), int(event["timing"]["available_frame"]),
        int(payload["closing_game_index_unverified"]),
        int(payload["opening_game_index_unverified"]),
    )


def _game_index_at_sequence(
    boundaries: Sequence[BoundaryEvidenceV1], sequence: int,
) -> int:
    sequences = [item.sequence for item in boundaries]
    return bisect_right(sequences, sequence - 1)


def _validate_candidate_game_indices(
    candidates: Sequence[PhysicalCandidateEvidenceV1],
    boundaries: Sequence[BoundaryEvidenceV1],
) -> None:
    for candidate in candidates:
        expected = _game_index_at_sequence(boundaries, candidate.sequence)
        if candidate.game_idx != expected:
            raise ValueError("物理着地候補と正式境界の局所試合番号が一致しません")


def _group_by_game_side(items: Sequence[Any]) -> dict[tuple[int, str], list[Any]]:
    result: dict[tuple[int, str], list[Any]] = defaultdict(list)
    for item in items:
        result[(item.game_idx, item.recipient)].append(item)
    return result


def _allocation_usage(
    matches: Sequence[ExchangeCrossMatchV1],
    supplies: Sequence[SentSupplyEvidenceV1],
) -> tuple[int, int]:
    used: Counter[str] = Counter()
    for match in matches:
        for allocation in match.allocations:
            used[allocation.sent_event_id] += allocation.amount
    capacities = {item.event_id: item.amount for item in supplies}
    overuse = sum(amount > capacities.get(event_id, 0) for event_id, amount in used.items())
    return sum(used.values()), overuse


def _candidate_by_id(
    candidates: Sequence[PhysicalCandidateEvidenceV1], event_id: str,
) -> PhysicalCandidateEvidenceV1:
    return next(item for item in candidates if item.event_id == event_id)


def _occurrence_overlaps(
    candidate: PhysicalCandidateEvidenceV1, supply: SentSupplyEvidenceV1,
) -> bool:
    return (
        supply.occurred_earliest_frame <= candidate.occurred_latest_frame
        and candidate.occurred_earliest_frame <= supply.occurred_latest_frame
    )


def _is_garbage_only(differences: Mapping[str, object], amount: int) -> bool:
    return (
        differences.get("color") == 0
        and differences.get("unknown") == 0
        and differences.get("occupied") == amount
    )


def _occupancy_corroborates(payload: Mapping[str, Any], amount: int) -> bool:
    raw = payload.get("raw_differences")
    if not isinstance(raw, Mapping) or raw.get("occupied") != amount:
        return False
    values = [raw.get(name) for name in ("color", "garbage", "unknown")]
    return all(_is_integer(value) for value in values) and sum(values) == amount


def _is_high(evidence: str) -> bool:
    return evidence in HIGH_CONFIDENCE_EVIDENCE


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_integer(value: object) -> bool:
    return _is_integer(value) and value > 0


__all__ = [
    "ExchangeCrossValidationReportV1",
    "audit_exchange_cross_quarantine_v1",
    "cross_validate_exchange_events_v1",
    "online_uncertain_quarantine_starts_v1",
    "unsupported_high_confidence_game_indices_v1",
    "unsupported_high_confidence_quarantine_starts_v1",
]
