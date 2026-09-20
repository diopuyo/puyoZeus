"""盤面から見たおじゃま着地と内部会計の落下量を独立に照合する。"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.event_accounting_adapter_v1 import (
    EventAccountingSidecar,
    load_event_accounting_sidecar,
)
from src.event_physical_adapter_v1 import (
    EventPhysicalSidecar,
    load_event_physical_sidecar,
)
from src.event_physical_observer_v1 import PhysicalObservationRow


VALIDATOR_VERSION = "event-physical-accounting-validator/v1"
SIDES = ("p1", "p2")


class PhysicalAccountingValidationError(ValueError):
    """同じ処理窓として安全に照合できない場合の例外。"""


@dataclass(frozen=True, slots=True)
class ModeledFall:
    """内部会計が一回の落下として減算した量。"""

    segment_index: int
    side: str
    ordinal: int
    frame_idx: int
    amount: int


@dataclass(frozen=True, slots=True)
class ObservedLanding:
    """状態遷移と前後盤面から得た一回の着地比較。"""

    segment_index: int
    side: str
    ordinal: int
    start_frame: int
    end_frame: int
    comparison_state: str
    observed_garbage_difference: int | None
    raw_comparison_state: str | None = None
    raw_garbage_difference: int | None = None
    confirmed_raw_agreement: bool | None = None


def validate_physical_accounting_files(
    physical_path: Path, accounting_path: Path,
) -> dict[str, Any]:
    """二つのサイドカーを厳格に読み、量の一致・不一致・欠測を返す。"""
    physical = load_event_physical_sidecar(physical_path)
    accounting = load_event_accounting_sidecar(accounting_path)
    return compare_physical_and_accounting(physical, accounting)


def compare_physical_and_accounting(
    physical: EventPhysicalSidecar, accounting: EventAccountingSidecar,
) -> dict[str, Any]:
    """同じ処理窓の落下を試合区間・side・発生順で一対一照合する。"""
    _validate_same_window(physical, accounting)
    boundaries = tuple(
        row.frame_idx for row in accounting.rows if row.formal_boundary
    )
    modeled = _modeled_falls(accounting, boundaries)
    observed = _observed_landings(physical, boundaries)
    comparisons = _pair_by_segment_side(modeled, observed)
    comparisons = _annotate_context(
        comparisons, physical.processing_start_frame, boundaries,
    )
    statuses = Counter(str(item["status"]) for item in comparisons)
    usable = [
        item for item in comparisons
        if item["association_state"] == "temporal_overlap"
        and item["status"].startswith("amount_")
    ]
    return {
        "validator_version": VALIDATOR_VERSION,
        "structural_validation_pass": True,
        "quantity_gate_pass": bool(comparisons) and all(
            item["status"] == "amount_exact:temporal_overlap"
            and item["context_state"] == "complete_since_boundary_or_video_start"
            for item in comparisons
        ),
        "processing_start_frame": physical.processing_start_frame,
        "processing_end_frame_exclusive": physical.processing_end_frame_exclusive,
        "formal_boundary_count": len(boundaries),
        "physical_counters": dict(sorted(physical.counters.items())),
        "modeled_fall_count": len(modeled),
        "modeled_fall_amount_total": sum(item.amount for item in modeled),
        "observed_landing_count": len(observed),
        "usable_amount_comparison_count": len(usable),
        "status_counts": dict(sorted(statuses.items())),
        "observed_exact_amount_total": sum(
            int(item["observed_amount"]) for item in usable
        ),
        "modeled_amount_in_usable_pairs": sum(
            int(item["modeled_amount"]) for item in usable
        ),
        "comparisons": comparisons,
    }


def _validate_same_window(
    physical: EventPhysicalSidecar, accounting: EventAccountingSidecar,
) -> None:
    physical_range = (
        physical.processing_start_frame, physical.processing_end_frame_exclusive,
    )
    accounting_range = (
        accounting.processing_start_frame, accounting.processing_end_frame_exclusive,
    )
    if physical_range != accounting_range:
        raise PhysicalAccountingValidationError("物理観測と会計の処理窓が違います")
    if physical.observed_frame_count != accounting.observed_frame_count:
        raise PhysicalAccountingValidationError("物理観測と会計の観測フレーム数が違います")
    if physical.inspected_side_count != accounting.inspected_side_count:
        raise PhysicalAccountingValidationError("物理観測と会計の検査side数が違います")


def _modeled_falls(
    accounting: EventAccountingSidecar, boundaries: Sequence[int],
) -> tuple[ModeledFall, ...]:
    ordinals: dict[tuple[int, str], int] = defaultdict(int)
    result: list[ModeledFall] = []
    for row in accounting.rows:
        segment = bisect_right(boundaries, row.frame_idx)
        for side in SIDES:
            amount = row.deltas[f"dropped_uncapped_{side}"]
            if amount <= 0:
                continue
            key = (segment, side)
            ordinals[key] += 1
            result.append(ModeledFall(segment, side, ordinals[key], row.frame_idx, amount))
    return tuple(result)


def _observed_landings(
    physical: EventPhysicalSidecar, boundaries: Sequence[int],
) -> tuple[ObservedLanding, ...]:
    completed = (
        row for row in physical.rows
        if row.observation_type == "landing_board_compared"
    )
    ordinals: dict[tuple[int, str], int] = defaultdict(int)
    result: list[ObservedLanding] = []
    for row in completed:
        start = int(row.payload["fall_start_frame"])
        segment = bisect_right(boundaries, start)
        key = (segment, row.side)
        ordinals[key] += 1
        result.append(_observed_landing(row, segment, ordinals[key]))
    return tuple(result)


def _observed_landing(
    row: PhysicalObservationRow, segment: int, ordinal: int,
) -> ObservedLanding:
    state = str(row.payload["comparison_state"])
    difference = row.payload.get("differences")
    amount = None
    if state == "consistent_garbage_addition" and isinstance(difference, Mapping):
        amount = int(difference["garbage"])
    return ObservedLanding(
        segment, row.side, ordinal, int(row.payload["fall_start_frame"]),
        row.frame_idx, state, amount,
        _optional_text(row.payload.get("raw_comparison_state")),
        _nested_integer(row.payload.get("raw_differences"), "garbage"),
        _optional_bool(row.payload.get("confirmed_raw_agreement")),
    )


def _pair_by_segment_side(
    modeled: Sequence[ModeledFall], observed: Sequence[ObservedLanding],
) -> list[dict[str, Any]]:
    modeled_groups = _group(modeled)
    observed_groups = _group(observed)
    keys = sorted(set(modeled_groups) | set(observed_groups))
    result: list[dict[str, Any]] = []
    for key in keys:
        left, right = modeled_groups[key], observed_groups[key]
        result.extend(_comparison(a, b) for a, b in _nearest_pairs(left, right))
    return result


def _nearest_pairs(
    modeled: Sequence[ModeledFall], observed: Sequence[ObservedLanding],
) -> list[tuple[ModeledFall | None, ObservedLanding | None]]:
    """時系列順を壊さず、少ない側を時間距離最小の候補へ対応させる。"""
    if not modeled:
        return [(None, item) for item in observed]
    if not observed:
        return [(item, None) for item in modeled]
    if len(modeled) <= len(observed):
        selected = _minimum_subsequence(
            [item.frame_idx for item in modeled],
            [item.start_frame for item in observed],
        )
        by_observed = {index: modeled[pos] for pos, index in enumerate(selected)}
        return [(by_observed.get(index), item) for index, item in enumerate(observed)]
    selected = _minimum_subsequence(
        [item.start_frame for item in observed],
        [item.frame_idx for item in modeled],
    )
    by_modeled = {index: observed[pos] for pos, index in enumerate(selected)}
    return [(item, by_modeled.get(index)) for index, item in enumerate(modeled)]


def _minimum_subsequence(targets: Sequence[int], candidates: Sequence[int]) -> list[int]:
    """targetsと同数の候補を順序維持で選び、絶対時間差の総和を最小化する。"""
    count, available = len(targets), len(candidates)
    costs = [[float("inf")] * (available + 1) for _ in range(count + 1)]
    take = [[False] * (available + 1) for _ in range(count + 1)]
    for index in range(available + 1):
        costs[0][index] = 0.0
    for target in range(1, count + 1):
        for candidate in range(target, available + 1):
            skipped = costs[target][candidate - 1]
            chosen = costs[target - 1][candidate - 1] + abs(
                targets[target - 1] - candidates[candidate - 1]
            )
            if chosen < skipped:
                costs[target][candidate], take[target][candidate] = chosen, True
            else:
                costs[target][candidate] = skipped
    return _trace_subsequence(take, count, available)


def _trace_subsequence(
    take: Sequence[Sequence[bool]], target: int, candidate: int,
) -> list[int]:
    selected: list[int] = []
    while target > 0:
        if take[target][candidate]:
            selected.append(candidate - 1)
            target -= 1
        candidate -= 1
    return list(reversed(selected))


def _group(items: Iterable[Any]) -> dict[tuple[int, str], list[Any]]:
    result: dict[tuple[int, str], list[Any]] = defaultdict(list)
    for item in items:
        result[(item.segment_index, item.side)].append(item)
    return result


def _annotate_context(
    comparisons: list[dict[str, Any]], start: int, boundaries: Sequence[int],
) -> list[dict[str, Any]]:
    left_unknown = start > 0 and (not boundaries or boundaries[0] > start)
    for item in comparisons:
        item["context_state"] = (
            "processing_left_context_unverified"
            if left_unknown and item["segment_index"] == 0
            else "complete_since_boundary_or_video_start"
        )
    return comparisons


def _comparison(
    modeled: ModeledFall | None, observed: ObservedLanding | None,
) -> dict[str, Any]:
    status = _comparison_status(modeled, observed)
    return {
        "segment_index": _first(modeled, observed, "segment_index"),
        "side": _first(modeled, observed, "side"),
        "ordinal": _first(modeled, observed, "ordinal"),
        "status": status,
        "modeled_frame": None if modeled is None else modeled.frame_idx,
        "modeled_amount": None if modeled is None else modeled.amount,
        "physical_start_frame": None if observed is None else observed.start_frame,
        "physical_end_frame": None if observed is None else observed.end_frame,
        "physical_comparison_state": (
            None if observed is None else observed.comparison_state
        ),
        "observed_amount": (
            None if observed is None else observed.observed_garbage_difference
        ),
        "raw_comparison_state": (
            None if observed is None else observed.raw_comparison_state
        ),
        "raw_observed_amount_diagnostic": (
            None if observed is None else observed.raw_garbage_difference
        ),
        "confirmed_raw_agreement": (
            None if observed is None else observed.confirmed_raw_agreement
        ),
        "association_state": _association_state(modeled, observed),
        "association_distance_frames": _association_distance(modeled, observed),
        "amount_difference_observed_minus_modeled": _amount_difference(modeled, observed),
    }


def _comparison_status(
    modeled: ModeledFall | None, observed: ObservedLanding | None,
) -> str:
    if modeled is None:
        return "physical_without_modeled_fall"
    if observed is None:
        return "modeled_fall_without_physical_landing"
    overlap = observed.start_frame <= modeled.frame_idx <= observed.end_frame
    if observed.observed_garbage_difference is None:
        if not overlap:
            return f"unverified_nearest_candidate:{observed.comparison_state}"
        return f"physical_amount_unusable:{observed.comparison_state}"
    amount = "exact" if observed.observed_garbage_difference == modeled.amount else "mismatch"
    if not overlap:
        return f"unverified_nearest_candidate:amount_{amount}"
    return f"amount_{amount}:temporal_overlap"


def _first(left: Any | None, right: Any | None, attribute: str) -> Any:
    source = left if left is not None else right
    if source is None:
        raise PhysicalAccountingValidationError("空の照合組が生成されました")
    return getattr(source, attribute)


def _amount_difference(
    modeled: ModeledFall | None, observed: ObservedLanding | None,
) -> int | None:
    if modeled is None or observed is None:
        return None
    if observed.observed_garbage_difference is None:
        return None
    return observed.observed_garbage_difference - modeled.amount


def _association_state(
    modeled: ModeledFall | None, observed: ObservedLanding | None,
) -> str:
    if modeled is None or observed is None:
        return "unpaired"
    if observed.start_frame <= modeled.frame_idx <= observed.end_frame:
        return "temporal_overlap"
    return "nearest_temporal_candidate_unverified"


def _association_distance(
    modeled: ModeledFall | None, observed: ObservedLanding | None,
) -> int | None:
    if modeled is None or observed is None:
        return None
    if modeled.frame_idx < observed.start_frame:
        return observed.start_frame - modeled.frame_idx
    if modeled.frame_idx > observed.end_frame:
        return modeled.frame_idx - observed.end_frame
    return 0


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _nested_integer(value: Any, key: str) -> int | None:
    if not isinstance(value, Mapping):
        return None
    item = value.get(key)
    return item if isinstance(item, int) and not isinstance(item, bool) else None


__all__ = [
    "PhysicalAccountingValidationError",
    "VALIDATOR_VERSION",
    "compare_physical_and_accounting",
    "validate_physical_accounting_files",
]
