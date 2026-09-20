"""実保存候補＋人工full-update正本の境界検査。実lockの認証ではない。"""
from __future__ import annotations
import copy
from dataclasses import replace
import hashlib
import json
from typing import Any
import numpy as np
import pytest
import torch
import binding as B
from src.advantage_m0_current_cnn_v1 import AdvantageM0CurrentCNNV2, AdvantageM0OutputV1
from src.advantage_m1_zero_counterfactual_v3 import AdvantageM1ZeroCounterfactualV3

SOURCE = B.ROOT.parent / "video38_current_scope_live_2026-09-09_v1/provisional_current.jsonl"
SOURCE_SHA = "4667bc54d161657851d62a87d1bd69fbe86aaa2772d511e0ea397afd33b66e13"
FRAME = 32684


@pytest.fixture(scope="module")
def candidates() -> dict[str, Any]:
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    return {row["side"]: row for row in rows if row["frame"] == FRAME}


def side_fixture(candidate: dict[str, Any]) -> dict[str, Any]:
    pb, sm, final, next_row = json.loads(candidate["current_candidate"]["_evidence_json"])
    before = {**final, "state": "STABLE", "state_value": "stable", "inferred": None,
        "probability": pb["probability"], "next_pair": pb["next_pair"], "dnext_pair": None}
    identity = dict.fromkeys(("isolate_return_is_final", "before_probability_matches_pb",
        "before_probability_unchanged", "after_probability_unchanged", "final_probability_matches_after"), True)
    return {"before_hold": before, "after_hold": copy.deepcopy(before), "final": copy.deepcopy(before),
        "pb": pb, "sm": sm, "next": next_row, "candidate_row": candidate, "identity": identity, "hold_reasons": []}


@pytest.fixture
def fixture(candidates: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sides = {side: side_fixture(copy.deepcopy(candidates[side])) for side in B.SIDES}
    identity = {key: sides["1P"]["pb"][key] for key in ("source_id", "run_id", "frame_idx", "time_sec")}
    row = {**identity, "schema_version": B.SCHEMA, "available_frame": FRAME, "capture_token": "artificial-update:1",
        "failures": [], "errors": [], "capture_status": "CAPTURED", "sides": sides,
        "update": {"returned": True, "exception": None,
        "call_index": 1, "pipe_index": 0, "frame_idx": FRAME, "returned_frame_idx": FRAME,
        "time_sec": identity["time_sec"], "returned_time_sec": identity["time_sec"],
        "is_match_active": True, "match_end_locked": False, "post_match_lockdown_active": False,
        "is_match_active_observed": True, "match_end_locked_observed": True, "post_match_lockdown_active_observed": True},
        "canonical": {"connection": "NOT_CONNECTED", "value": None, "reason": "producer_not_connected"},
        "ledger": {"connection": "NOT_CONNECTED", "sides": {side: {field:
        {"value": None, "availability": "UNKNOWN", "reason": "producer_not_connected"}
        for field in B.LEDGER_SIDE_FIELDS} for side in B.SIDES}}}
    row["game"] = {"observed": False, "value": None, "reason": "no_game_identity_on_this_pipeline_boundary"}
    generation = {"observed": True, "reason": "actual_software_generation_not_physical_game", "value": {side:
        {"side": side, "reset_epoch": 2, "action_revision": 1,
         "identity_scope": "software_observation_only_not_physical_identity"} for side in B.SIDES}}
    row["generation"] = {"before": generation, "after": copy.deepcopy(generation)}
    row["upstream_failures"] = dict.fromkeys(("hidden_probability_observer", "current_scope_sink",
                                             "provisional_current_connection"), [])
    registration = {key: identity[key] for key in ("source_id", "run_id")}
    registration.update(time_base_numerator=1, time_base_denominator=60, ledger_connection="NOT_CONNECTED")
    return row, registration


def test_unknown_typed_inputs_and_real_m1_m0_bit_copy(fixture: Any) -> None:
    bound = B.bind_provisional_context(*fixture)
    assert not bound.supported and bound.integrity_valid
    assert np.all(bound.inputs.queues[:, 2:] == 0)
    torch.manual_seed(930)
    model = AdvantageM1ZeroCounterfactualV3(AdvantageM0CurrentCNNV2(), "values_and_masks").eval()
    actual = B.evaluate_bound(bound, model, sample_count=8)
    with torch.no_grad():
        expected = model.m0(torch.tensor(bound.inputs.boards[None], dtype=torch.int64),
                            torch.tensor(bound.inputs.queues[None], dtype=torch.int64)).raw_probability.item()
    assert actual.win_probability_p1 == expected
    assert not any(value.flags.writeable for value in (bound.inputs.boards, bound.inputs.queues,
        bound.inputs.ledger_values, bound.inputs.ledger_availability))


def test_held_published_grid_does_not_become_model_current(fixture: Any) -> None:
    row, registration = fixture
    side = row["sides"]["2P"]
    for key in ("after_hold", "final"):
        side[key]["confirmed"] = None
        side[key]["probability"] = {"present": False, "type_valid": False, "cells": None,
            "errors": ["missing_probability"], "sha256": None}
        side[key]["board_none_reason"] = B.HELD_REASON
    side["candidate_row"]["confirmed_output_held"] = True
    bound = B.bind_provisional_context(row, registration)
    assert bound.inputs.boards[1].sum() > 0 and row["sides"]["2P"]["final"]["confirmed"] is None


@pytest.mark.parametrize("key,value", (("match_end_locked", True), ("match_end_locked", None),
    ("post_match_lockdown_active", True), ("post_match_lockdown_active", None), ("is_match_active", False)))
def test_hard_control_hold(fixture: Any, key: str, value: Any) -> None:
    fixture[0]["update"][key] = value
    with pytest.raises(B.ContextHold):
        B.bind_provisional_context(*fixture)


@pytest.mark.parametrize("path,value", (
    (("frame_idx",), True), (("time_sec",), float("nan")), (("available_frame",), FRAME + 2),
    (("run_id",), "other"), (("update", "returned_frame_idx"), FRAME + 2),
    (("update", "match_end_locked"), 0), (("update", "exception"), "failure"),
    (("update", "pipe_index"), False), (("failures",), ["capture failed"]),
    (("ledger", "connection"), "ERROR"), (("canonical", "connection"), "ERROR"),
    (("ledger", "sides", "1P", "pending_garbage", "availability"), "INTEGRITY_FAULT"),
    (("ledger", "sides", "1P", "pending_garbage", "value"), 0),
    (("sides", "1P", "identity", "isolate_return_is_final"), False),
    (("sides", "1P", "identity", "before_probability_unchanged"), False),
    (("sides", "1P", "final", "next_pair"), [1, 2]),
    (("sides", "1P", "next", "frame_idx"), FRAME + 2),
    (("sides", "1P", "sm", "software_epoch"), True),
))
def test_fault_not_hidden_as_unknown(fixture: Any, path: tuple[str, ...], value: Any) -> None:
    target = fixture[0]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(B.ContextFault):
        B.bind_provisional_context(*fixture)


@pytest.mark.parametrize("value", ([True, 2], [1.0, 2], ["1", 2], [11, 2]))
def test_dnext_strict_before_shared_normalizer(fixture: Any, value: Any) -> None:
    for stage in ("before_hold", "after_hold", "final"):
        fixture[0]["sides"]["1P"][stage]["dnext_pair"] = value
    with pytest.raises(B.ContextFault, match="queue_type"):
        B.bind_provisional_context(*fixture)


def test_input_and_control_tamper_rejected_without_model_call(fixture: Any) -> None:
    bound = B.bind_provisional_context(*fixture)
    for changed in (replace(bound, supported=True), replace(bound, integrity_valid=False),
                    replace(bound, context_digest="f" * 64)):
        with pytest.raises(B.ContextFault, match="bound_tampered"):
            B.evaluate_bound(changed, None)
    copied = bound.inputs.queues.copy()
    copied[:] = 0
    changed = replace(bound, inputs=replace(bound.inputs, queues=copied))
    with pytest.raises(B.ContextFault, match="bound_inputs_changed"):
        B.evaluate_bound(changed, None)


def test_bound_source_detached_and_run_in_digest(fixture: Any) -> None:
    bound = B.bind_provisional_context(*fixture)
    fixture[0]["update"]["match_end_locked"] = True
    assert json.loads(bound.source_json)["row"]["update"]["match_end_locked"] is False
    payload = json.loads(bound.source_json)
    payload["registration"]["run_id"] = "different"
    assert B.digest(payload) != bound.context_digest


@pytest.mark.parametrize("key", ("match_end_locked", "is_match_active", "post_match_lockdown_active"))
def test_unobserved_normal_value_is_not_truth(fixture: Any, key: str) -> None:
    fixture[0]["update"][key + "_observed"] = False
    with pytest.raises(B.ContextHold, match="unobserved"):
        B.bind_provisional_context(*fixture)


def test_real_software_reset_and_known_game_not_erased(fixture: Any) -> None:
    row, registration = fixture
    row["generation"]["after"]["value"]["1P"]["reset_epoch"] += 1
    with pytest.raises(B.ContextHold, match="generation_changed"):
        B.bind_provisional_context(row, registration)
    row["generation"]["after"] = copy.deepcopy(row["generation"]["before"])
    row["game"] = {"observed": True, "value": {"1P": 1, "2P": 2}, "reason": "actual"}
    with pytest.raises(B.ContextFault, match="game"):
        B.bind_provisional_context(row, registration)
