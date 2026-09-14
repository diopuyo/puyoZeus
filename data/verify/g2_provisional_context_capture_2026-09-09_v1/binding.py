"""同update暫定contextを既存M1の未対応分岐へ束縛する診断用入口。"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "g2_generation_publication_split_2026-09-09_v1"
sys.path[:0] = [str(PRIOR), str(ROOT.parents[2])]
import current_policy as C
from src.advantage_m0_current_cnn_v1 import board_categories_from_raw, queue_categories_from_raw
from src.advantage_m1_causal_ledger_v3 import (
    AdvantageM1PrimaryV3, AdvantageM1PrimarySideV3, PrimaryIntV3, PrimaryBoolV3,
    AdvantageM1InputsV3, tensorize_primary, LEDGER_SIDE_FIELDS, AVAILABILITY_ORDER,
)
from src.canonical_observation_v3 import AvailabilityState

SIDES = ("1P", "2P")
ROWS, COLS, QUEUE_SLOTS = 13, 6, 4
RAW_COLORS = frozenset((0, 1, 2, 3, 4, 5, 9, 10))
QUEUE_COLORS = frozenset((0, 1, 2, 3, 4, 5, 9, 10))
HELD_REASON = "discarded_candidate_current_unverified"
STAGE = "before_confirmed_publication_hold"
SCHEMA = "provisional-current-context/v1"
BRIDGE = ROOT.parent / "g2_hidden_probability_provisional_2026-09-08_v1/model_bridge.py"
BRIDGE_SHA = "d7f1a0658aff0a8bc42386d1377e320a5835aa8d9b9ee418e9e24ce0328f1022"


class ContextHold(ValueError):
    """通常の非適格・欠測。モデルを呼ばず保留する。"""


class ContextFault(ValueError):
    """来歴・型・計装の矛盾。UNKNOWNへ救済しない。"""


def require(condition: bool, reason: str, *, hold: bool = False) -> None:
    if not condition:
        raise (ContextHold if hold else ContextFault)(reason)


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def exact(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return bool(left == right)


def integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _identity(row: dict[str, Any], registration: dict[str, Any]) -> None:
    require(type(row) is dict and row.get("schema_version") == SCHEMA, "context_schema")
    require(type(registration) is dict, "registration_type")
    for key in ("source_id", "run_id"):
        require(type(row.get(key)) is str and bool(row[key])
                and exact(row[key], registration.get(key)), "context_identity:" + key)
    frame, clock = row.get("frame_idx"), row.get("time_sec")
    require(integer(frame) and type(clock) in (int, float) and math.isfinite(clock) and clock >= 0,
            "context_clock_type")
    numerator, denominator = registration.get("time_base_numerator"), registration.get("time_base_denominator")
    require(integer(numerator) and numerator > 0 and integer(denominator) and denominator > 0, "time_base")
    require(clock == frame * numerator / denominator, "context_time_base")
    require(integer(row.get("available_frame")) and row["available_frame"] == frame, "context_available")
    require(type(row.get("capture_token")) is str and bool(row["capture_token"]), "capture_token")
    require(set(row.get("sides", {})) == set(SIDES), "context_sides")
    require(row.get("failures") == [], "context_capture_failure")
    require(row.get("capture_status") == "CAPTURED", "context_capture_status")
    expected_upstream = dict.fromkeys(("hidden_probability_observer", "current_scope_sink",
                                      "provisional_current_connection"), [])
    require(exact(row.get("upstream_failures"), expected_upstream), "context_upstream_failure")
    require(registration.get("ledger_connection") == "NOT_CONNECTED", "registration_ledger_connection")


def _update(row: dict[str, Any]) -> None:
    update = row.get("update")
    require(type(update) is dict and update.get("returned") is True and update.get("exception") is None,
            "update_not_returned")
    for key in ("call_index", "pipe_index"):
        require(integer(update.get(key)), "update_identity:" + key)
    for key in ("frame_idx", "returned_frame_idx"):
        require(integer(update.get(key)) and update[key] == row["frame_idx"], "update_frame")
    for key in ("time_sec", "returned_time_sec"):
        require(exact(update.get(key), row["time_sec"]), "update_time")
    for key, expected in (("is_match_active", True), ("match_end_locked", False),
                          ("post_match_lockdown_active", False)):
        require(update.get(key) is None or type(update[key]) is bool, "hard_control_type:" + key)
        require(update.get(key + "_observed") is True, "hard_control_unobserved:" + key, hold=True)
        require(update.get(key) is expected, "hard_control_hold:" + key, hold=True)


def _generation(row: dict[str, Any]) -> None:
    require(exact(row.get("game"), {"observed": False, "value": None,
        "reason": "no_game_identity_on_this_pipeline_boundary"}), "unsupported_or_contradictory_game")
    slots = row.get("generation")
    require(type(slots) is dict and set(slots) == {"before", "after"}, "generation_slots")
    for slot in slots.values():
        require(type(slot) is dict and slot.get("observed") is True, "generation_unobserved", hold=True)
        require(slot.get("reason") == "actual_software_generation_not_physical_game", "generation_semantics")
        values = slot.get("value")
        require(type(values) is dict and set(values) == set(SIDES), "generation_sides")
        for side in SIDES:
            value = values[side]
            require(type(value) is dict and value.get("side") == side, "generation_side")
            require(integer(value.get("reset_epoch")) and (value.get("action_revision") is None
                or integer(value["action_revision"])), "generation_integer")
            require(value.get("identity_scope") == "software_observation_only_not_physical_identity", "generation_scope")
    require(exact(slots["before"], slots["after"]), "generation_changed_inside_update", hold=True)


def _raw_grid(value: Any) -> np.ndarray:
    require(type(value) is dict, "grid_missing")
    grid = value.get("grid")
    require(type(grid) is list and len(grid) == ROWS, "grid_rows")
    require(all(type(row) is list and len(row) == COLS for row in grid), "grid_columns")
    require(all(type(color) is int and color in RAW_COLORS for row in grid for color in row), "grid_colors")
    return np.asarray(grid, dtype=np.int64)


def _queue(value: Any) -> list[int | None]:
    if value is None:
        return [None, None]
    require(type(value) is list and len(value) == 2, "queue_shape")
    require(all(color is None or type(color) is int and color in QUEUE_COLORS for color in value), "queue_type")
    return value.copy()


def _probability(value: Any) -> list[Any]:
    require(type(value) is dict and value.get("present") is True
            and value.get("type_valid") is True and value.get("errors") == [], "probability_invalid")
    cells = value.get("cells")
    require(type(cells) is list and len(cells) == ROWS, "probability_rows")
    return cells


def _published(side: dict[str, Any], pb: dict[str, Any]) -> None:
    before, after, final = (side.get(key) for key in ("before_hold", "after_hold", "final"))
    require(all(type(value) is dict for value in (before, after, final)), "publication_stage_missing")
    require(all(value.get("state") == "STABLE" for value in (before, after, final)), "non_stable", hold=True)
    require(before.get("board_none_reason") is None and before.get("board_provenance") == "observed", "before_not_observed")
    require(exact(before.get("confirmed"), pb.get("confirmed")), "before_pb_grid_mismatch")
    require(exact(_probability(before.get("probability")), _probability(pb.get("probability"))), "before_pb_distribution")
    require(exact(after, final), "full_update_changed_after_isolate")
    identity = side.get("identity", {})
    for key in ("isolate_return_is_final", "before_probability_matches_pb", "before_probability_unchanged",
                "after_probability_unchanged", "final_probability_matches_after"):
        require(identity.get(key) is True, "object_or_distribution_identity:" + key)
    for key in ("cnn", "inferred", "next_pair", "dnext_pair", "board_provenance"):
        require(exact(before.get(key), after.get(key)), "unexpected_publication_change:" + key)
    if after.get("confirmed") is None:
        require(after.get("board_none_reason") == HELD_REASON, "unexplained_publication_hold")
        require(type(after.get("probability")) is dict and after["probability"].get("present") is False,
                "held_probability_not_none")
    else:
        require(exact(before, after), "normal_publication_changed")


def _candidate(row: dict[str, Any], side_name: str) -> Any:
    side = row["sides"][side_name]
    require(type(side) is dict and side.get("hold_reasons") == [], "side_capture_hold", hold=True)
    candidate = side.get("candidate_row")
    require(type(candidate) is dict and candidate.get("current_stage") == STAGE, "candidate_stage")
    value = candidate.get("current_candidate")
    require(type(value) is dict, "current_candidate_missing", hold=True)
    require(candidate.get("frame") == row["frame_idx"] and type(candidate.get("frame")) is int
            and candidate.get("side") == side_name, "candidate_scope")
    require(candidate.get("accounting_permission") is False and candidate.get("quality_gate_clear") is False,
            "candidate_permission_changed")
    evidence = json.loads(value.get("_evidence_json", "null"))
    require(type(evidence) is list and len(evidence) == 4, "candidate_evidence")
    pb, sm, final, next_row = evidence
    for saved, key in ((pb, "pb"), (sm, "sm"), (next_row, "next")):
        require(exact(saved, side.get(key)), "candidate_saved_join:" + key)
    for key in ("source_id", "run_id", "frame_idx", "time_sec"):
        require(exact(pb.get(key), row[key]), "candidate_context_identity:" + key)
    require(pb.get("side") == side_name, "candidate_side")
    try:
        restored = C.current_candidate(*evidence)
    except ValueError as error:
        raise ContextHold("current_policy:" + str(error)) from error
    require(encoded(asdict(restored)) == encoded(value), "candidate_dto_changed")
    before = side.get("before_hold", {})
    for key in ("confirmed", "cnn", "board_provenance", "board_none_reason"):
        require(exact(final.get(key), before.get(key)), "candidate_before_stage:" + key)
    require(exact(before.get("next_pair"), pb.get("next_pair")), "candidate_before_queue")
    _published(side, pb)
    require(candidate.get("confirmed_output_held") is (side["after_hold"]["confirmed"] is None),
            "candidate_hold_stage_mismatch")
    return restored


def _unknown_ledger(row: dict[str, Any]) -> tuple[Any, Any]:
    canonical, ledger = row.get("canonical"), row.get("ledger")
    require(exact(canonical, {"connection": "NOT_CONNECTED", "value": None,
        "reason": "producer_not_connected"}), "canonical_connection")
    require(type(ledger) is dict and ledger.get("connection") == "NOT_CONNECTED", "ledger_connection")
    require(set(ledger.get("sides", {})) == set(SIDES), "ledger_sides")
    result = []
    for side in SIDES:
        fields = ledger["sides"][side]
        require(type(fields) is dict and set(fields) == set(LEDGER_SIDE_FIELDS), "ledger_fields")
        for value in fields.values():
            require(exact(value, {"value": None, "availability": "UNKNOWN", "reason": "producer_not_connected"}),
                    "unconnected_ledger_not_unknown")
        values = [PrimaryBoolV3(None, AvailabilityState.UNKNOWN) if name == "chain_active"
                  else PrimaryIntV3(None, AvailabilityState.UNKNOWN) for name in LEDGER_SIDE_FIELDS]
        result.append(AdvantageM1PrimarySideV3(*values))
    return tuple(result)


def _inputs(row: dict[str, Any]) -> AdvantageM1InputsV3:
    boards, queues = [], []
    for side in SIDES:
        before = row["sides"][side]["before_hold"]
        boards.append(board_categories_from_raw(_raw_grid(before["confirmed"])))
        values = _queue(before.get("next_pair")) + _queue(before.get("dnext_pair"))
        require(len(values) == QUEUE_SLOTS, "queue_slots")
        queues.append(queue_categories_from_raw(values))
    primary = AdvantageM1PrimaryV3(np.stack(boards), np.stack(queues), *_unknown_ledger(row))
    result = tensorize_primary(primary)
    require(bool(np.isfinite(result.ledger_values).all()) and bool((result.ledger_values == 0).all()), "unknown_values")
    masks = result.ledger_availability
    require(bool(np.isin(masks, (0, 1)).all()) and bool((masks.sum(axis=-1) == 1).all()), "availability_one_hot")
    require(bool((masks[..., AVAILABILITY_ORDER.index(AvailabilityState.UNKNOWN)] == 1).all()), "availability_unknown")
    for value in (result.boards, result.queues, result.ledger_values, result.ledger_availability):
        value.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class BoundContext:
    """計算入力と全正本を保持。再bindしてから評価する。"""
    inputs: AdvantageM1InputsV3
    context_digest: str
    source_json: str
    supported: bool = False
    integrity_valid: bool = True
    reason: str = "provisional_current_ledger_not_connected_m0_only"


def bind_provisional_context(row: dict[str, Any], registration: dict[str, Any]) -> BoundContext:
    """保存SHA・coverage確認は呼出側。canonical認証やM1 residual許可は発行しない。"""
    _identity(row, registration)
    _update(row)
    _generation(row)
    for side in SIDES:
        _candidate(row, side)
    inputs = _inputs(row)
    payload = {"row": row, "registration": registration}
    return BoundContext(inputs, digest(payload), encoded(payload))


def evaluate_bound(bound: BoundContext, model: Any, *, sample_count: int = 256, seed: int = 0) -> Any:
    """callerのboolではなく、検査済み一次contextから制御値を再導出する。"""
    require(type(bound) is BoundContext, "bound_type")
    payload = json.loads(bound.source_json)
    fresh = bind_provisional_context(payload["row"], payload["registration"])
    require(bound.context_digest == fresh.context_digest and bound.supported is False
            and bound.integrity_valid is True and bound.reason == fresh.reason, "bound_tampered")
    for key in ("boards", "queues", "ledger_values", "ledger_availability"):
        require(np.array_equal(getattr(bound.inputs, key), getattr(fresh.inputs, key)), "bound_inputs_changed")
    require(hashlib.sha256(BRIDGE.read_bytes()).hexdigest() == BRIDGE_SHA, "model_bridge_changed")
    spec = importlib.util.spec_from_file_location("_context_probability_bridge", BRIDGE)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    scorer = bridge.M1ProbabilityScorer(model, fresh.inputs, supported=fresh.supported,
                                        integrity_valid=fresh.integrity_valid)
    pair = [_candidate(payload["row"], side) for side in SIDES]
    return C.evaluate_pair(*pair, scorer, sample_count=sample_count, seed=seed)
