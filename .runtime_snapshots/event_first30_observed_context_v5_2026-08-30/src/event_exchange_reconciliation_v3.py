"""確定会計と盤面着地を分離して照合する交換監査v3。"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.event_accounting_adapter_v1 import EventAccountingSidecar
from src.event_physical_adapter_v1 import EventPhysicalSidecar
from src.event_physical_observer_v1 import PhysicalObservationRow


SIDES = ("p1", "p2")
OTHER_SIDE = {"p1": "p2", "p2": "p1"}


@dataclass(frozen=True, slots=True)
class ModeledSettlementV3:
    """確定会計上の落下予約または境界消滅。"""

    settlement_id: str
    frame_idx: int
    game_idx: int
    recipient: str
    settlement_type: str
    amount: int


@dataclass(frozen=True, slots=True)
class PhysicalLandingCandidateV3:
    """確定盤面差で正方向に見えたおじゃま量。"""

    frame_idx: int
    game_idx: int
    recipient: str
    visible_amount: int
    evidence: str


@dataclass(frozen=True, slots=True)
class SettlementAllocationV3:
    """物理着地候補へ割り当てた確定会計量。"""

    settlement_id: str
    settlement_frame: int
    settlement_type: str
    amount: int


@dataclass(frozen=True, slots=True)
class PhysicalLandingMatchV3:
    """一つの物理候補と同一試合・同一受け手の確定会計との照合。"""

    frame_idx: int
    game_idx: int
    recipient: str
    visible_amount: int
    evidence: str
    matched_amount: int
    unexplained_amount: int
    confirmation_available_frame: int | None
    timing_state: str
    status: str
    allocations: tuple[SettlementAllocationV3, ...]


@dataclass(frozen=True, slots=True)
class ExchangeReconciliationReportV3:
    """確定残高と物理着地を混ぜない交換監査結果。"""

    total_generated: int
    total_canceled: int
    total_sent: int
    total_reserved: int
    total_boundary_expired: int
    total_final_pending: int
    attack_conservation_residual: int
    pending_conservation_residual_p1: int
    pending_conservation_residual_p2: int
    provisional_finalization_count: int
    provisional_corrected_count: int
    provisional_missing_count: int
    all_clear_gained_count: int
    all_clear_consumed_count: int
    physical_candidate_count: int
    physical_candidate_amount: int
    physical_evidence_counts: dict[str, int]
    explained_candidate_count: int
    unexplained_candidate_count: int
    unexplained_candidate_amount: int
    high_confidence_candidate_count: int
    high_confidence_unexplained_count: int
    low_confidence_unexplained_count: int
    late_confirmed_candidate_count: int
    max_confirmation_lag_frames: int
    physical_gate_pass: bool
    accounting_gate_pass: bool
    all_pass: bool
    settlements: tuple[ModeledSettlementV3, ...]
    matches: tuple[PhysicalLandingMatchV3, ...]

    def to_json_value(self) -> dict[str, Any]:
        """監査成果物へ保存できるJSON互換値を返す。"""
        value = asdict(self)
        value["schema_version"] = "event-exchange-reconciliation/v3"
        return value


def reconcile_event_exchange_v3(
    accounting: EventAccountingSidecar,
    physical: EventPhysicalSidecar,
) -> ExchangeReconciliationReportV3:
    """確定会計を保存則へ、盤面差を独立した着地検算へ使う。"""
    _validate_matching_ranges(accounting, physical)
    boundaries = tuple(row.frame_idx for row in accounting.rows if row.formal_boundary)
    settlements = _modeled_settlements(accounting, boundaries)
    candidates = _physical_candidates(physical)
    matches = _match_candidates(candidates, settlements)
    accounting_totals = _accounting_totals(accounting)
    provisional = _provisional_audit(accounting)
    evidence = Counter(candidate.evidence for candidate in candidates)
    unexplained = [match for match in matches if match.unexplained_amount > 0]
    high_candidates = [item for item in candidates if _is_high_confidence(item.evidence)]
    high_unexplained = [
        item for item in unexplained if _is_high_confidence(item.evidence)
    ]
    lags = [
        match.confirmation_available_frame - match.frame_idx
        for match in matches
        if match.confirmation_available_frame is not None
        and match.confirmation_available_frame > match.frame_idx
    ]
    accounting_pass = all(
        accounting_totals[key] == 0 for key in (
            "attack_residual", "pending_residual_p1", "pending_residual_p2",
        )
    )
    physical_pass = not high_unexplained
    return ExchangeReconciliationReportV3(
        accounting_totals["generated"], accounting_totals["canceled"],
        accounting_totals["sent"], accounting_totals["reserved"],
        accounting_totals["expired"], accounting_totals["final_pending"],
        accounting_totals["attack_residual"],
        accounting_totals["pending_residual_p1"],
        accounting_totals["pending_residual_p2"],
        provisional["finalizations"], provisional["corrected"],
        provisional["missing"], _physical_count(physical, "all_clear_gained_candidate"),
        _physical_count(physical, "all_clear_consumed_candidate"), len(candidates),
        sum(item.visible_amount for item in candidates), dict(sorted(evidence.items())),
        len(matches) - len(unexplained), len(unexplained),
        sum(item.unexplained_amount for item in unexplained),
        len(high_candidates), len(high_unexplained),
        len(unexplained) - len(high_unexplained),
        sum(item.timing_state == "confirmed_after_observation" for item in matches),
        max(lags, default=0), physical_pass, accounting_pass,
        physical_pass and accounting_pass, settlements, matches,
    )


def _accounting_totals(sidecar: EventAccountingSidecar) -> dict[str, int]:
    generated = sum(attack.generated_amount for row in sidecar.rows for attack in row.attacks)
    canceled = sum(
        row.deltas[f"offset_uncapped_{side}"] for row in sidecar.rows for side in SIDES
    )
    sent = generated - canceled
    reserved = sum(
        row.deltas[f"dropped_uncapped_{side}"] for row in sidecar.rows for side in SIDES
    )
    expired = sum(
        row.deltas[f"boundary_wiped_uncapped_{side}"]
        for row in sidecar.rows for side in SIDES
    )
    residuals = _pending_residuals(sidecar)
    return {
        "generated": generated, "canceled": canceled, "sent": sent,
        "reserved": reserved, "expired": expired,
        "final_pending": sum(sidecar.final_pending),
        "attack_residual": generated - canceled - sent,
        "pending_residual_p1": residuals[0], "pending_residual_p2": residuals[1],
    }


def _pending_residuals(sidecar: EventAccountingSidecar) -> tuple[int, int]:
    values: list[int] = []
    for index, recipient in enumerate(SIDES):
        sender = OTHER_SIDE[recipient]
        sent = sum(
            attack.generated_amount - row.deltas[f"offset_uncapped_{sender}"]
            for row in sidecar.rows for attack in row.attacks if attack.side == sender
        )
        consumed = sum(
            row.deltas[f"offset_uncapped_{recipient}"]
            + row.deltas[f"dropped_uncapped_{recipient}"]
            + row.deltas[f"boundary_wiped_uncapped_{recipient}"]
            for row in sidecar.rows
        )
        values.append(sent - consumed - sidecar.final_pending[index])
    return values[0], values[1]


def _provisional_audit(sidecar: EventAccountingSidecar) -> dict[str, int]:
    latest: dict[tuple[str, int], int] = {}
    finalizations = corrected = missing = 0
    for row in sidecar.rows:
        for item in row.provisional_updates:
            latest[(item.side, item.resolver_chain_ordinal)] = (
                item.provisional_generated_amount
            )
        for attack in row.attacks:
            finalizations += 1
            provisional = latest.get((attack.side, attack.resolver_chain_ordinal))
            if provisional is None:
                missing += 1
            elif provisional != attack.generated_amount:
                corrected += 1
    return {"finalizations": finalizations, "corrected": corrected, "missing": missing}


def _modeled_settlements(
    sidecar: EventAccountingSidecar, boundaries: Sequence[int],
) -> tuple[ModeledSettlementV3, ...]:
    result: list[ModeledSettlementV3] = []
    ordinal = 0
    for row in sidecar.rows:
        game_idx = bisect_right(boundaries, row.frame_idx)
        for side in SIDES:
            for event_type, key in (
                ("fall_reserved", f"dropped_uncapped_{side}"),
                ("boundary_expired", f"boundary_wiped_uncapped_{side}"),
            ):
                amount = row.deltas[key]
                if amount <= 0:
                    continue
                ordinal += 1
                result.append(ModeledSettlementV3(
                    f"settlement-{ordinal:06d}", row.frame_idx, game_idx,
                    side, event_type, amount,
                ))
    return tuple(result)


def _physical_candidates(
    sidecar: EventPhysicalSidecar,
) -> tuple[PhysicalLandingCandidateV3, ...]:
    result: list[PhysicalLandingCandidateV3] = []
    for row in sidecar.rows:
        if row.observation_type != "landing_board_compared":
            continue
        candidate = _physical_candidate(row)
        if candidate is not None:
            result.append(candidate)
    return tuple(result)


def _physical_candidate(
    row: PhysicalObservationRow,
) -> PhysicalLandingCandidateV3 | None:
    payload = row.payload
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
    return PhysicalLandingCandidateV3(
        row.frame_idx, row.game_idx, row.side, amount, evidence,
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


def _match_candidates(
    candidates: Sequence[PhysicalLandingCandidateV3],
    settlements: Sequence[ModeledSettlementV3],
) -> tuple[PhysicalLandingMatchV3, ...]:
    grouped_candidates = _group_by_game_side(candidates)
    grouped_settlements = _group_by_game_side(settlements)
    matches: list[PhysicalLandingMatchV3] = []
    for key in sorted(grouped_candidates):
        capacities = {item.settlement_id: item.amount for item in grouped_settlements[key]}
        candidates_by_priority = sorted(
            grouped_candidates[key], key=lambda item: (
                0 if _is_high_confidence(item.evidence) else 1, item.frame_idx,
            ),
        )
        for candidate in candidates_by_priority:
            matches.append(_match_one(
                candidate, grouped_settlements[key], capacities,
            ))
    return tuple(sorted(matches, key=lambda item: (item.frame_idx, item.recipient)))


def _match_one(
    candidate: PhysicalLandingCandidateV3,
    settlements: Sequence[ModeledSettlementV3],
    capacities: dict[str, int],
) -> PhysicalLandingMatchV3:
    remaining = candidate.visible_amount
    allocations: list[SettlementAllocationV3] = []
    while remaining > 0:
        available = [item for item in settlements if capacities[item.settlement_id] > 0]
        if not available:
            break
        selected = min(
            available,
            key=lambda item: (abs(item.frame_idx - candidate.frame_idx), item.frame_idx),
        )
        amount = min(remaining, capacities[selected.settlement_id])
        capacities[selected.settlement_id] -= amount
        remaining -= amount
        allocations.append(SettlementAllocationV3(
            selected.settlement_id, selected.frame_idx,
            selected.settlement_type, amount,
        ))
    confirmation = max((item.settlement_frame for item in allocations), default=None)
    timing = _timing_state(candidate.frame_idx, confirmation, remaining)
    status = "explained" if remaining == 0 else "unexplained"
    return PhysicalLandingMatchV3(
        candidate.frame_idx, candidate.game_idx, candidate.recipient,
        candidate.visible_amount, candidate.evidence,
        candidate.visible_amount - remaining, remaining, confirmation,
        timing, status, tuple(allocations),
    )


def _group_by_game_side(items: Sequence[Any]) -> dict[tuple[int, str], list[Any]]:
    result: dict[tuple[int, str], list[Any]] = defaultdict(list)
    for item in items:
        side = item.recipient
        result[(item.game_idx, side)].append(item)
    for values in result.values():
        values.sort(key=lambda item: item.frame_idx)
    return result


def _timing_state(frame: int, confirmation: int | None, remaining: int) -> str:
    if remaining > 0 or confirmation is None:
        return "unexplained"
    if confirmation > frame:
        return "confirmed_after_observation"
    return "available_at_observation"


def _physical_count(sidecar: EventPhysicalSidecar, event_type: str) -> int:
    return sum(row.observation_type == event_type for row in sidecar.rows)


def _is_high_confidence(evidence: str) -> bool:
    return evidence in {"raw_exact", "raw_occupancy_corroborated"}


def _validate_matching_ranges(
    accounting: EventAccountingSidecar, physical: EventPhysicalSidecar,
) -> None:
    accounting_range = (
        accounting.processing_start_frame, accounting.processing_end_frame_exclusive,
    )
    physical_range = (
        physical.processing_start_frame, physical.processing_end_frame_exclusive,
    )
    if accounting_range != physical_range:
        raise ValueError("会計と物理観測の処理範囲が一致しません")


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_integer(value: object) -> bool:
    return _is_integer(value) and value > 0


__all__ = [
    "ExchangeReconciliationReportV3",
    "PhysicalLandingMatchV3",
    "reconcile_event_exchange_v3",
]
