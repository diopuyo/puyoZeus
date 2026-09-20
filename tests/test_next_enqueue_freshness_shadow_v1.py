"""NEXT案AのCPU境界反例。全update/物理着手/同色着手の合格ではない。"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import next_enqueue_freshness_shadow_v1 as subject


@pytest.fixture(scope="module")
def blocks() -> subject.base.FrozenAccountingBlocks:
    return subject.base.FrozenAccountingBlocks.load()


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    value = subject.base._read_verified(subject.base.RUN / "frames.jsonl", subject.base.RUN_SHA)
    return [json.loads(line) for line in value.splitlines()]


def clock(frame: int) -> subject.base.Clock:
    return subject.base.Clock(frame, frame / subject.base.FPS)


def evidence(frame: int, side: str = "1P", **changes: Any) -> subject.SlideEvidence:
    value = subject.SlideEvidence(side, clock(frame), False, 1.0, 8.0, 2, True, None,
                                  str(subject.base.PIPELINE), subject.CALL_LINES[side])
    return replace(value, **changes)


def packet(frame: int, *, side: str = "1P", pair: tuple[int, int] | None = (5, 5),
           before: str = "menu", after: str = "menu", **changes: Any) -> subject.NextObservation:
    value = subject.NextObservation(side, clock(frame), True, pair, evidence(frame, side), before, after)
    return replace(value, **changes)


def owner(blocks: subject.base.FrozenAccountingBlocks, side: str = "1P") -> subject.NextEnqueueOwner:
    return subject.NextEnqueueOwner(side, blocks)


def saved_packet(frame_rows: list[dict[str, Any]], side: str, before: str) -> subject.NextObservation:
    accounting = next(row for row in frame_rows if row["kind"] == "accounting_update" and row["side"] == side)
    state = next(row for row in frame_rows if row["kind"] == "frame_side" and row["side"] == side)
    main = next((row for row in frame_rows if row["kind"] == "initial_pair_main_next"), None)
    slide = next((row for row in frame_rows if row["kind"] == "initial_pair_main_slide" and row["side"] == side), None)
    pair = None if main is None else tuple(main["returned"][side]["next"])
    return subject.NextObservation(side, clock(accounting["frame_idx"]), accounting["is_match_active"],
                                   pair, None if slide is None else subject.SlideEvidence.from_saved_row(slide),
                                   before, state["state"])


@pytest.mark.parametrize("side", subject.base.SIDES)
def test_real_saved_stream_keeps_true_pairs_before_landings(
    blocks: subject.base.FrozenAccountingBlocks, rows: list[dict[str, Any]], side: str,
) -> None:
    instance, state, events = owner(blocks, side), "menu", []
    for frame in range(32656, 32784, 2):
        observation = saved_packet([row for row in rows if row["frame_idx"] == frame], side, state)
        raw_before = repr(observation)
        result = instance.process(observation)
        assert result.observation is observation and repr(observation) == raw_before
        assert not result.live_permission and not result.physical_placement_verified
        assert result.input_authentication_scope == subject.INPUT_SCOPE
        if result.enqueued is not None:
            events.append((frame, result.enqueued))
        state = observation.after_state
    assert events == [(32704, (5, 5)), (32754, (3, 5))]
    assert instance.snapshot.accounting.counts == ((3, 1), (5, 3))
    assert instance.snapshot.accounting.pending == ()
    assert instance.snapshot.accepted_next == (5, 5)
    expected_frame = 32730 if side == "1P" else 32728
    assert instance.snapshot.accounting.first_move_sec == expected_frame / subject.base.FPS


def test_original_ast_matches_all_128_saved_side_rows(rows: list[dict[str, Any]]) -> None:
    path = subject.base.ROOT / "data/verify/next_enqueue_freshness_cpu_2026-09-08_v1/probe.py"
    subject.base._read_verified(path, "70271b1c74cf24707f7626b5b5227f6b1612fe6a8f7562c7251dd05b1f25026c")
    spec = importlib.util.spec_from_file_location("next_cpu_original_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    methods = module.original_methods(module.verified())
    result = module.replay(rows, "old", methods)
    assert result["old_saved_mismatches"] == []
    assert len(result["enqueues"]) == 8


def test_known_four_are_high_diff_false_and_not_accepted(rows: list[dict[str, Any]]) -> None:
    selected = [row for row in rows if row["kind"] == "initial_pair_main_slide"
                and row["frame_idx"] in (32698, 32702)]
    assert len(selected) == 4
    for row in selected:
        slide = subject.SlideEvidence.from_saved_row(row)
        assert slide.diff_score > slide.threshold_used and slide.cooldown_after > 0 and not slide.pulse
        observation = packet(row["frame_idx"], side=row["side"], slide=slide)
        assert subject._reason(observation) == "moving_next_rejected"


@pytest.mark.parametrize("change", [{"diff_score": 60.0}, {"diff_score": 8.0},
                                    {"pulse": True}, {"actual_call": False}])
def test_unsafe_slide_does_not_replace_accounting_history(
    blocks: subject.base.FrozenAccountingBlocks, change: dict[str, Any],
) -> None:
    instance = owner(blocks)
    instance.process(packet(32666))
    result = instance.process(packet(32668, pair=(2, 3), slide=evidence(32668, **change)))
    assert result.enqueued is None and result.after.accepted_next == (5, 5)
    assert not result.after.accounting.pending


def test_quiet_during_cooldown_is_accepted_without_zero_wait(blocks: subject.base.FrozenAccountingBlocks) -> None:
    instance = owner(blocks)
    instance.process(packet(32666))
    result = instance.process(packet(32668, pair=(3, 5), slide=evidence(32668, cooldown_after=5)))
    assert result.enqueued == (5, 5) and result.after.accepted_next == (3, 5)


@pytest.mark.parametrize("field,value", [("diff_score", float("nan")), ("diff_score", float("inf")),
                                        ("diff_score", True), ("diff_score", -1), ("threshold_used", 0),
                                        ("threshold_used", float("nan"))])
def test_invalid_slide_numeric_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        evidence(32666, **{field: value})


@pytest.mark.parametrize("changes", [{"side": "2P"}, {"episode_id": "other"},
                                     {"source_sha256": "bad"}, {"run_sha256": "bad"}])
def test_observation_identity_rejected_without_owner_change(
    blocks: subject.base.FrozenAccountingBlocks, changes: dict[str, Any],
) -> None:
    instance = owner(blocks)
    before = instance._bundle
    with pytest.raises(ValueError):
        instance.process(replace(packet(32666), **changes))
    assert instance._bundle is before


@pytest.mark.parametrize("slide", [evidence(32668), evidence(32666, side="2P"),
                                   evidence(32666, caller_line=1), evidence(32666, caller_source="other.py")])
def test_cross_frame_side_or_caller_slide_is_rejected(
    blocks: subject.base.FrozenAccountingBlocks, slide: subject.SlideEvidence,
) -> None:
    instance = owner(blocks)
    before = instance._bundle
    with pytest.raises(ValueError):
        instance.process(packet(32666, slide=slide))
    assert instance._bundle is before


@pytest.mark.parametrize("pair", [(0, 5), (10, 5), (9, 5), None])
def test_unknown_next_is_not_enqueued(blocks: subject.base.FrozenAccountingBlocks, pair: Any) -> None:
    instance = owner(blocks)
    result = instance.process(packet(32666, pair=pair))
    assert result.after.accepted_next is None and result.enqueued is None


def test_actual_exception_null_return_is_preserved_without_zero_fill(
    blocks: subject.base.FrozenAccountingBlocks,
) -> None:
    row = dict(kind="initial_pair_main_slide", frame_idx=32668, time_sec=32668 / 60,
               side="1P", extra_call=False, exception="RuntimeError:fixture", returned=None,
               after={"cooldown": 1}, caller={"source": str(subject.base.PIPELINE), "line": 5077})
    slide = subject.SlideEvidence.from_saved_row(row)
    assert slide.diff_score is None and slide.threshold_used is None and slide.pulse is None
    instance = owner(blocks)
    instance.process(packet(32666))
    result = instance.process(packet(32668, pair=(2, 3), slide=slide))
    assert result.reason == "slide_call_failed" and result.after.accepted_next == (5, 5)
    assert result.enqueued is None
    with pytest.raises(ValueError):
        subject.SlideEvidence.from_saved_row({**row, "exception": None})


def test_missing_call_not_quiet_and_raw_objects_unchanged(blocks: subject.base.FrozenAccountingBlocks) -> None:
    instance = owner(blocks)
    observation = packet(32666, slide=None)
    result = instance.process(observation)
    assert result.reason == "slide_not_observed" and result.after.accepted_next is None
    assert result.observation.next_pair == (5, 5)
    with pytest.raises(FrozenInstanceError):
        observation.next_pair = (4, 4)
    with pytest.raises(ValueError):
        packet(32668, pair=[4, 4])


def test_same_frame_replay_is_same_receipt_and_changed_payload_rejected(
    blocks: subject.base.FrozenAccountingBlocks,
) -> None:
    instance = owner(blocks)
    observation = packet(32666)
    result = instance.process(observation)
    before = instance._bundle
    assert instance.process(replace(observation)) is result
    with pytest.raises(ValueError):
        instance.process(replace(observation, next_pair=(3, 5)))
    assert instance._bundle is before


def test_reverse_clock_and_state_discontinuity_refused(blocks: subject.base.FrozenAccountingBlocks) -> None:
    instance = owner(blocks)
    instance.process(packet(32666))
    before = instance._bundle
    for observation in (packet(32664), packet(32668, before="stable")):
        with pytest.raises(ValueError):
            instance.process(observation)
    assert instance._bundle is before


def test_missing_frame_is_not_bridged_by_old_accepted_pair(blocks: subject.base.FrozenAccountingBlocks) -> None:
    instance = owner(blocks)
    instance.process(packet(32666))
    before = instance._bundle
    with pytest.raises(ValueError, match="欠落"):
        instance.process(packet(32670, pair=(3, 5)))
    assert instance._bundle is before


def test_inactive_clear_and_next_seed_are_not_initial_owner_retention(
    blocks: subject.base.FrozenAccountingBlocks,
) -> None:
    instance = owner(blocks)
    instance.process(packet(32666))
    instance.process(packet(32668, pair=(3, 5)))
    assert instance.snapshot.accounting.pending == ((5, 5),)
    result = instance.process(packet(32670, active=False, pair=None, slide=None))
    assert result.after.accepted_next is None and not result.after.accounting.pending
    assert not result.after.accounting.counts and result.after.accounting.first_move_sec is None
    result = instance.process(packet(32672, pair=(3, 5)))
    assert result.enqueued is None and result.after.accepted_next == (3, 5)


def test_reset_clears_and_rejects_old_epoch(blocks: subject.base.FrozenAccountingBlocks) -> None:
    instance = owner(blocks)
    instance.process(packet(32666))
    instance.reset("cpu:explicit-next-epoch", clock(32668))
    before = instance._bundle
    with pytest.raises(ValueError):
        instance.process(packet(32670))
    assert instance._bundle is before
    result = instance.process(packet(32670, episode_id="cpu:explicit-next-epoch"))
    assert result.enqueued is None
    with pytest.raises(ValueError):
        instance.reset("cpu:explicit-next-epoch", clock(32672))


def test_same_color_value_does_not_create_a_physical_event(blocks: subject.base.FrozenAccountingBlocks) -> None:
    instance = owner(blocks)
    first = instance.process(packet(32666, pair=(4, 4)))
    second = instance.process(packet(32668, pair=(4, 4)))
    assert first.enqueued is None and second.enqueued is None
    assert not second.same_color_placement_resolved and not second.physical_placement_verified


def prepare_pending(instance: subject.NextEnqueueOwner) -> None:
    instance.process(packet(32666))
    instance.process(packet(32668, pair=(3, 5)))
    instance.process(packet(32670, pair=(3, 5), after="tsumo_fall"))


def test_partial_color_fault_preserves_whole_bundle_then_retry_once(
    blocks: subject.base.FrozenAccountingBlocks, monkeypatch: Any,
) -> None:
    instance = owner(blocks)
    prepare_pending(instance)
    observation = packet(32672, pair=(3, 5), before="tsumo_fall", after="stable")
    before = instance._bundle
    class FailSecondColor(Counter):
        def __setitem__(self, key: int, value: int) -> None:
            if value == 2:
                raise RuntimeError("人工:片色後失敗")
            super().__setitem__(key, value)
    with monkeypatch.context() as patch:
        patch.setattr(subject, "Counter", FailSecondColor)
        with pytest.raises(RuntimeError):
            instance.process(observation)
    assert instance._bundle is before and not instance._busy
    result = instance.process(observation)
    assert result.after.accounting.counts == ((5, 2),)
    assert result.after.accounting.first_move_sec == 32672 / 60
    assert instance.process(observation) is result


def test_reentry_is_refused_before_any_bundle_swap(blocks: subject.base.FrozenAccountingBlocks, monkeypatch: Any) -> None:
    instance = owner(blocks)
    before = instance._bundle
    observation = packet(32666)
    original = blocks.replay
    def reenter(name: str, namespace: dict[str, Any]) -> None:
        instance.process(observation)
        original(name, namespace)
    with monkeypatch.context() as patch:
        patch.setattr(blocks, "replay", reenter)
        with pytest.raises(ValueError, match="再入"):
            instance.process(observation)
    assert instance._bundle is before


def test_reset_fault_does_not_clear_owner_then_throw(blocks: subject.base.FrozenAccountingBlocks, monkeypatch: Any) -> None:
    instance = owner(blocks)
    instance.process(packet(32666))
    before = instance._bundle
    original = blocks.replay
    def fail(name: str, namespace: dict[str, Any]) -> None:
        original(name, namespace)
        raise RuntimeError("人工:private clear後")
    with monkeypatch.context() as patch:
        patch.setattr(blocks, "replay", fail)
        with pytest.raises(RuntimeError):
            instance.reset("cpu:new", clock(32668))
    assert instance._bundle is before


def test_normal_fifo_and_first_move_not_reassigned(blocks: subject.base.FrozenAccountingBlocks) -> None:
    instance = owner(blocks)
    prepare_pending(instance)
    first = instance.process(packet(32672, pair=(3, 5), before="tsumo_fall", after="stable"))
    instance.process(packet(32674, pair=(5, 5), before="stable", after="tsumo_fall"))
    second = instance.process(packet(32676, pair=(5, 5), before="tsumo_fall", after="stable"))
    assert second.after.accounting.counts == ((3, 1), (5, 3))
    assert not second.after.accounting.pending
    assert second.after.accounting.first_move_sec == first.after.accounting.first_move_sec


def test_original_collector_drain_is_once_per_consumed_pair(blocks: subject.base.FrozenAccountingBlocks) -> None:
    path = subject.base.FROZEN / "scripts/collect_boards_lean.py"
    tree = ast.parse(subject.base._read_verified(path, subject.base.REQUIRED_INPUT_SHA256[str(path)]))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                and item.name == "_drain_ojama_by_tsumo_delta_lean")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    program = compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), str(path), "exec")
    namespace: dict[str, Any] = {}
    exec(program, namespace)
    calls: list[tuple[str, float]] = []
    tracker = SimpleNamespace(on_tsumo_settled=lambda side, time: calls.append((side, time)))
    drain_state = SimpleNamespace(ojama_prev_tsumo=0)
    instance = owner(blocks)
    prepare_pending(instance)
    observation = packet(32672, pair=(3, 5), before="tsumo_fall", after="stable")
    result = instance.process(observation)
    for receipt in (result, instance.process(observation)):
        total = sum(count for _, count in receipt.after.accounting.counts) // 2
        namespace[node.name](tracker, "p1", drain_state, total, observation.clock.time_sec)
    assert calls == [("p1", observation.clock.time_sec)] and drain_state.ojama_prev_tsumo == 1


def test_source_guards_no_live_api_and_fifty_line_functions() -> None:
    assert not hasattr(subject, "install") and not hasattr(subject, "apply_to_pipeline")
    for path, expected in subject.REQUIRED_INPUT_SHA256.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    for path in (Path(subject.__file__), Path(__file__)):
        nodes = ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        assert all(node.end_lineno - node.lineno + 1 <= 50
                   for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
