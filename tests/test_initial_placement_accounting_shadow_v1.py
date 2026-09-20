"""初手会計CPU部品。人工境界fixtureと固定独立oracleをlive証明と混同しない。"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter, deque
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import initial_placement_accounting_shadow_v1 as subject


@pytest.fixture(scope="module")
def blocks() -> subject.FrozenAccountingBlocks:
    return subject.FrozenAccountingBlocks.load()


@pytest.fixture(scope="module")
def oracle_template() -> subject.CpuReviewOracle:
    return subject.CpuReviewOracle.load()


def oracle_clone(template: subject.CpuReviewOracle) -> subject.CpuReviewOracle:
    """重い再hashを省く人工fixture。同一proof/ownerの共有はしない。"""
    oracle = object.__new__(subject.CpuReviewOracle)
    oracle._observations = template._observations
    oracle._proofs = {side: subject.CpuLandingProof(side) for side in subject.SIDES}
    oracle._owners = {}
    return oracle


@pytest.fixture
def oracle(oracle_template: subject.CpuReviewOracle) -> subject.CpuReviewOracle:
    return oracle_clone(oracle_template)


def clock(frame: int) -> subject.Clock:
    return subject.Clock(frame, frame / subject.FPS)


def ready(oracle: subject.CpuReviewOracle, side: str = "1P", *,
          landing_activity: bool = True) -> subject.InitialPlacementOwner:
    owner = subject.InitialPlacementOwner(oracle, side)
    for observation in oracle.observations(side):
        owner.observe(observation)
    if landing_activity:
        owner.observe_activity(clock(subject.LANDING_FRAME), True)
    return owner


def context(side: str = "1P") -> subject.LandingContext:
    return subject.LandingContext(side, clock(subject.LANDING_FRAME))


def pipeline_fixture() -> SimpleNamespace:
    """凍結会計blockの私有境界fixture。実pipelineを生成しない。"""
    pipe = SimpleNamespace()
    for side in ("1p", "2p"):
        values = {"tsumo_count": Counter(), "pending_tsumo": deque(),
                  "last_seen_next": None, "first_move_sec": None,
                  "constraint_valid": True, "landing_pending": None,
                  "last_consumed_color": None}
        for field, value in values.items():
            setattr(pipe, f"_{field}_{side}", value)
    return pipe


def enqueue(blocks: subject.FrozenAccountingBlocks, pipe: Any, frame: int,
            pair1: tuple[int, int], pair2: tuple[int, int]) -> None:
    blocks.replay("enqueue", {"self": pipe, "frame_idx": frame, "is_active": True,
                              "next_pair_1p": pair1, "next_pair_2p": pair2})


def settle(blocks: subject.FrozenAccountingBlocks, pipe: Any, side: str,
           frame: int, before: subject.CpuState) -> None:
    blocks.replay("settle", {"self": pipe, "side": side, "prev_state": before,
                             "ctx": SimpleNamespace(state=subject.CpuState.STABLE),
                             "signals": SimpleNamespace(time_sec=frame / subject.FPS)})


def test_frozen_before_matches_real_saved_boundary_rows(blocks: subject.FrozenAccountingBlocks) -> None:
    rows = [json.loads(line) for line in (subject.RUN / "frames.jsonl").read_text().splitlines()]
    actual = {(row["frame_idx"], row["side"]): row for row in rows if row["kind"] == "accounting_update"}
    pipe = pipeline_fixture()
    enqueue(blocks, pipe, 32656, (5, 5), (5, 5))
    blocks.replay("clear", {"self": pipe})
    assert pipe._last_seen_next_1p is None
    enqueue(blocks, pipe, 32666, (5, 5), (5, 5))
    for side in subject.SIDES:
        settle(blocks, pipe, side, 32684, subject.CpuState.MENU)
        assert dict(getattr(pipe, f"_tsumo_count_{side.lower()}")) == actual[32684, side]["after"]["tsumo_count"] == {}
    enqueue(blocks, pipe, 32698, (2, 3), (1, 3))
    enqueue(blocks, pipe, 32702, (3, 5), (3, 5))
    for side, frame in (("2P", 32728), ("1P", 32730)):
        settle(blocks, pipe, side, frame, subject.CpuState.TSUMO_FALL)
        value = getattr(pipe, f"_tsumo_count_{side.lower()}")
        assert {str(key): count for key, count in value.items()} == actual[frame, side]["after"]["tsumo_count"] == {"5": 2}
        assert getattr(pipe, f"_first_move_sec_{side.lower()}") == actual[frame, side]["after"]["first_move_sec"]


def test_pending_alone_does_not_fix_menu_landing(blocks: subject.FrozenAccountingBlocks) -> None:
    pipe = pipeline_fixture()
    pipe._pending_tsumo_1p.append((4, 4))
    settle(blocks, pipe, "1P", 32684, subject.CpuState.MENU)
    assert pipe._tsumo_count_1p == {} and list(pipe._pending_tsumo_1p) == [(4, 4)]


@pytest.mark.parametrize("side", subject.SIDES)
def test_reviewed_replay_consumes_once_without_state_action_spoof(
    oracle: subject.CpuReviewOracle, blocks: subject.FrozenAccountingBlocks, side: str,
) -> None:
    owner = ready(oracle, side, landing_activity=False)
    for frame, active in ((32656, True), (32658, False), (32666, True)):
        owner.observe_activity(clock(frame), active)
        assert owner.snapshot == subject.CpuAccountingSnapshot()
    owner.observe_activity(clock(subject.LANDING_FRAME), True)
    ctx = context(side)
    receipt = owner.consume(oracle.proof(side), ctx, blocks)
    assert receipt.after.counts == ((4, 2),) and not receipt.after.pending
    assert receipt.after.first_move_sec == ctx.clock.time_sec
    assert receipt.after.consumed_id.endswith(f":{side}:initial-placement")
    assert receipt.context is ctx and ctx.before_state is subject.CpuState.MENU
    assert ctx.action_revision is None and receipt.after.revision == 1
    assert not receipt.live_permission and not receipt.accounting_basis_verified
    before_retry = owner.snapshot
    with pytest.raises(ValueError):
        owner.consume(oracle.proof(side), ctx, blocks)
    assert owner.snapshot is before_retry


@pytest.mark.parametrize("frame,time", [(True, 1), (-1, -1), (32684, True),
                                       (32684, float("nan")), (32684, float("inf")), (32684, 1)])
def test_clock_rejects_bad_values(frame: Any, time: Any) -> None:
    with pytest.raises(ValueError):
        subject.Clock(frame, time)


@pytest.mark.parametrize("pair", [(0, 4), (4, 10), (4, 9), (True, 4), [4, 4]])
def test_observation_rejects_unknown_or_mutable_pair(pair: Any) -> None:
    with pytest.raises(ValueError):
        subject.PairObservation("1P", clock(32496), pair, (5, 5))


@pytest.mark.parametrize("field,value", [("side", "2P"), ("episode_id", "other"),
                                         ("source_sha256", "bad"), ("run_sha256", "bad")])
def test_cross_identity_observation_rejected(oracle: subject.CpuReviewOracle, field: str, value: Any) -> None:
    owner = subject.InitialPlacementOwner(oracle, "1P")
    with pytest.raises(ValueError):
        owner.observe(replace(oracle.observations("1P")[0], **{field: value}))


def test_same_frame_replay_and_collision(oracle: subject.CpuReviewOracle) -> None:
    owner = subject.InitialPlacementOwner(oracle, "1P")
    observation = oracle.observations("1P")[0]
    owner.observe(observation)
    owner.observe(replace(observation))
    assert len(owner._observations) == 1
    with pytest.raises(ValueError):
        owner.observe(replace(observation, next_pair=(1, 1)))


def test_clock_reverse_and_activity_collision(oracle: subject.CpuReviewOracle) -> None:
    owner = ready(oracle, landing_activity=False)
    owner.observe_activity(clock(32656), True)
    with pytest.raises(ValueError):
        owner.observe_activity(clock(32656), False)
    with pytest.raises(ValueError):
        owner.observe_activity(clock(32654), True)


def test_reset_invalidates_observation_without_counter_restore(
    oracle: subject.CpuReviewOracle, blocks: subject.FrozenAccountingBlocks,
) -> None:
    owner = ready(oracle, landing_activity=False)
    before = owner.snapshot
    owner.invalidate_for_reset(clock(32658))
    assert owner.snapshot is before and not owner._observations
    with pytest.raises(ValueError):
        owner.consume(oracle.proof("1P"), context(), blocks)


@pytest.mark.parametrize("proof_kind", ["none", "boolean", "copy", "other_oracle", "other_side"])
def test_unissued_proof_never_authorizes(
    oracle: subject.CpuReviewOracle, blocks: subject.FrozenAccountingBlocks, proof_kind: str,
) -> None:
    owner = ready(oracle)
    proofs = {"none": None, "boolean": True, "copy": replace(oracle.proof("1P")),
              "other_oracle": oracle_clone(oracle).proof("1P"), "other_side": oracle.proof("2P")}
    with pytest.raises(ValueError):
        owner.consume(proofs[proof_kind], context(), blocks)
    assert owner.snapshot == subject.CpuAccountingSnapshot()


@pytest.mark.parametrize("changes", [{"execution_scope": "live"}, {"episode_id": "other"},
                                     {"reset_frame": 32658}, {"before_state": subject.CpuState.TSUMO_FALL},
                                     {"clock": subject.Clock(32682, 32682 / 60)}, {"active": False},
                                     {"action_revision": 999}])
def test_wrong_context_rejected(
    oracle: subject.CpuReviewOracle, blocks: subject.FrozenAccountingBlocks, changes: dict[str, Any],
) -> None:
    owner = ready(oracle)
    with pytest.raises(ValueError):
        owner.consume(oracle.proof("1P"), replace(context(), **changes), blocks)


@pytest.mark.parametrize("active", [False, None])
def test_landing_activity_contradiction_or_missing_refused(
    oracle: subject.CpuReviewOracle, blocks: subject.FrozenAccountingBlocks, active: bool | None,
) -> None:
    owner = ready(oracle, landing_activity=False)
    if active is not None:
        owner.observe_activity(clock(subject.LANDING_FRAME), active)
    before = owner.snapshot
    with pytest.raises(ValueError, match="activity"):
        owner.consume(oracle.proof("1P"), context(), blocks)
    assert owner.snapshot is before


@pytest.mark.parametrize("snapshot", [subject.CpuAccountingSnapshot(counts=((5, 2),)),
                                      subject.CpuAccountingSnapshot(pending=((5, 5),)),
                                      subject.CpuAccountingSnapshot(first_move_sec=544.0),
                                      subject.CpuAccountingSnapshot(revision=1)])
def test_progressed_state_refuses_backfill(
    oracle: subject.CpuReviewOracle, blocks: subject.FrozenAccountingBlocks,
    snapshot: subject.CpuAccountingSnapshot,
) -> None:
    owner = ready(oracle)
    owner._state = snapshot  # 検査直前の所有状態変化を作る負例。
    with pytest.raises(ValueError):
        owner.consume(oracle.proof("1P"), context(), blocks)
    assert owner.snapshot is snapshot


def test_incomplete_observation_and_duplicate_owner_refused(
    oracle: subject.CpuReviewOracle, blocks: subject.FrozenAccountingBlocks,
) -> None:
    owner = subject.InitialPlacementOwner(oracle, "1P")
    owner.observe(oracle.observations("1P")[0])
    with pytest.raises(ValueError):
        owner.consume(oracle.proof("1P"), context(), blocks)
    with pytest.raises(ValueError):
        subject.InitialPlacementOwner(oracle, "1P")


def test_private_copy_exception_after_first_color_is_atomic(
    oracle: subject.CpuReviewOracle, blocks: subject.FrozenAccountingBlocks, monkeypatch: Any,
) -> None:
    owner = ready(oracle)
    before = owner.snapshot
    class FailSecondColor(Counter):
        def __setitem__(self, key: int, value: int) -> None:
            if value == 2:
                raise RuntimeError("人工:片色加算後の例外")
            super().__setitem__(key, value)
    with monkeypatch.context() as patch:
        patch.setattr(subject, "Counter", FailSecondColor)
        with pytest.raises(RuntimeError, match="人工"):
            owner.consume(oracle.proof("1P"), context(), blocks)
    assert owner.snapshot is before and not owner._busy
    assert owner.consume(oracle.proof("1P"), context(), blocks).after.counts == ((4, 2),)


def test_normal_fifo_same_color_distinct_hands_and_first_move_preserved(
    blocks: subject.FrozenAccountingBlocks,
) -> None:
    state = subject.CpuAccountingSnapshot(pending=((5, 5), (5, 5)), first_move_sec=544.7)
    first = blocks.consume_copy(state, "1P", clock(32730))
    second = blocks.consume_copy(first, "1P", clock(32782))
    assert first.counts == ((5, 2),) and first.pending == ((5, 5),)
    assert second.counts == ((5, 4),) and not second.pending
    assert second.first_move_sec == 544.7 and state.pending == ((5, 5), (5, 5))


def test_cooldown_enqueue_remains_an_explicit_unresolved_counterexample(
    blocks: subject.FrozenAccountingBlocks,
) -> None:
    pipe = pipeline_fixture()
    enqueue(blocks, pipe, 32666, (5, 5), (5, 5))
    enqueue(blocks, pipe, 32698, (2, 3), (1, 3))
    enqueue(blocks, pipe, 32702, (3, 5), (3, 5))
    assert list(pipe._pending_tsumo_1p) == [(5, 5), (2, 3)]
    assert list(pipe._pending_tsumo_2p) == [(5, 5), (1, 3)]
    rows = [json.loads(line) for line in (subject.RUN / "frames.jsonl").read_text().splitlines()]
    slides = [row for row in rows if row["kind"] == "initial_pair_main_slide"
              and row["frame_idx"] in (32698, 32702)]
    assert len(slides) == 4
    for row in slides:
        assert row["before"]["cooldown"] > 0 and not row["returned"]["slide_motion"]
        assert row["returned"]["diff_score"] > row["returned"]["threshold_used"]


def test_real_frozen_collector_drain_once_after_pair_consumption(
    oracle: subject.CpuReviewOracle, blocks: subject.FrozenAccountingBlocks,
) -> None:
    path = subject.FROZEN / "scripts/collect_boards_lean.py"
    tree = ast.parse(subject._read_verified(path, subject.REQUIRED_INPUT_SHA256[str(path)]))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "_drain_ojama_by_tsumo_delta_lean")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    program = compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), str(path), "exec")
    namespace: dict[str, Any] = {}
    exec(program, namespace)
    drain = namespace[node.name]
    calls: list[tuple[str, float]] = []
    tracker = SimpleNamespace(on_tsumo_settled=lambda side, time: calls.append((side, time)))
    state = SimpleNamespace(ojama_prev_tsumo=0)
    receipt = ready(oracle).consume(oracle.proof("1P"), context(), blocks)
    settled_pairs = sum(count for _, count in receipt.after.counts) // 2
    drain(tracker, "p1", state, settled_pairs, context().clock.time_sec)
    drain(tracker, "p1", state, settled_pairs, context().clock.time_sec)
    assert calls == [("p1", context().clock.time_sec)] and state.ojama_prev_tsumo == 1


def test_load_refuses_changed_evidence_without_real_file_edit(monkeypatch: Any) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", lambda self: b"synthetic mismatch")
        with pytest.raises(ValueError, match="SHA"):
            subject.CpuReviewOracle.load()


def test_receipt_cannot_request_live_permission() -> None:
    with pytest.raises(TypeError):
        subject.CpuConsumptionReceipt(subject.CpuAccountingSnapshot(),
                                      subject.CpuAccountingSnapshot(), context(), live_permission=True)


def test_no_live_entry_no_mutable_snapshots_and_frozen_inputs_unchanged() -> None:
    assert not hasattr(subject, "install") and not hasattr(subject, "apply_to_pipeline")
    with pytest.raises(FrozenInstanceError):
        subject.CpuAccountingSnapshot().revision = 2
    with pytest.raises(ValueError):
        subject.CpuAccountingSnapshot(pending=[(4, 4)])
    with pytest.raises(ValueError):
        subject.CpuReviewOracle(object(), [])
    with pytest.raises(ValueError):
        subject.FrozenAccountingBlocks(object(), {})
    for path, expected in subject.REQUIRED_INPUT_SHA256.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected


def test_functions_within_fifty_lines() -> None:
    for path in (Path(subject.__file__), Path(__file__)):
        nodes = ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        assert all(node.end_lineno - node.lineno + 1 <= 50
                   for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
