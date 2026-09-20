"""C-6 pending commit shadowの副作用遮断・復元・receipt検査。"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import diagnose_video38_c6_pending_commit_shadow_v1 as diag
from src import chain_prediction_ledger_v1 as ledger_module
from src import board_state_machine as board_state_module
from src.board import Board, COLOR_BLUE, COLOR_RED
from src.chain import ChainSimulator
from src.chain_detector import CHAIN_MECHANISM_LANDING, ChainEvent
from src.scoring import calculate_chain_score


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/diagnose_video38_c6_pending_commit_shadow_v1.py"
FRAME, TIME = 34864, 581.0666666666667


class _Generations:
    def __init__(self) -> None:
        self.value = {side: SimpleNamespace(side=side, reset_epoch=0, action_revision=25)
                      for side in ("1P", "2P")}
        self._in_frame, self._frame, self._time = True, FRAME, TIME

    def generation(self, side: str) -> Any:
        return self.value[side]

    def set(self, side: str, action: int) -> None:
        self.value[side] = SimpleNamespace(side=side, reset_epoch=0,
                                           action_revision=action)


def _chain_board(side: str = "2P") -> Board:
    grid = [[0] * 6 for _ in range(13)]
    column = 0 if side == "1P" else 5
    for row in range(9, 13):
        grid[row][column] = COLOR_RED
    return Board.from_list(grid)


def _observed_board() -> Board:
    grid = [[0] * 6 for _ in range(13)]
    grid[12][2] = COLOR_BLUE
    return Board.from_list(grid)


def _transform_passthrough(value: Any) -> Any:
    return value


def _event(board: Board, mechanism: str = CHAIN_MECHANISM_LANDING) -> ChainEvent:
    result = ChainSimulator().simulate(board)
    score = calculate_chain_score(result).total_score
    return ChainEvent(trigger_sec=578.366, end_sec=579.0, before_board=board,
        chain_count=max(1, result.chain_count), total_erased=result.total_erased,
        total_score=score, base_score=score, all_clear_bonus_applied=0,
        ojama_sent=0, leftover_score=0, is_all_clear=False,
        mechanism=mechanism, score_estimated=False)


class _Pipe:
    def __init__(self, event: ChainEvent, side: str = "2P") -> None:
        self._chain_sim = ChainSimulator()
        for suffix in ("1p", "2p"):
            setattr(self, f"_stable_color_memory_{suffix}", {(0, 0): COLOR_BLUE})
            setattr(self, f"_tsumo_count_{suffix}", Counter({COLOR_RED: 20}))
            setattr(self, f"_constraint_valid_{suffix}", False)
            setattr(self, f"_chain_verify_pending_{suffix}", None)
            setattr(self, f"_prev_stable_confirmed_{suffix}", _observed_board())
            setattr(self, f"_last_chain_event_for_settle_{suffix}", None)
        suffix = "1p" if side == "1P" else "2p"
        setattr(self, f"_last_chain_event_for_settle_{suffix}", event)


def _context() -> Any:
    board = _observed_board()
    return SimpleNamespace(state="stable", confirmed_board=board,
                           pending_board=board.copy())


def _recorder(side: str = "2P", open_landing: bool = True) -> tuple[Any, Any, io.StringIO]:
    stream = io.StringIO()
    rec = diag.PendingCommitRecorder(stream, {})
    generations = _Generations()
    rec.bind_runtime(ledger_module, generations)
    rec.bind_commit_runtime(board_state_module)
    rec.begin_frame(FRAME, TIME)
    if open_landing:
        board = _chain_board(side)
        rec.record_start(side, _event(board), ChainSimulator().simulate(board))
    return rec, generations, stream


def _run_c6(rec: Any, side: str = "2P") -> tuple[Any, Any]:
    board, ctx = _chain_board(side), _context()
    event = _event(board, "formula_read")
    pipe = _Pipe(event, side)
    rec.record_c6_pending(pipe, side, FRAME, TIME, ctx, event)
    return pipe, ctx


def _rows(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_c6_creates_one_pending_transaction_without_internal_mutation() -> None:
    rec, _generations, _stream = _recorder()
    pipe, ctx = _run_c6(rec)
    record = rec.active_candidates["2P"]
    assert record["transaction"].state.value == "pending"
    assert ctx.confirmed_board == _observed_board() and ctx.pending_board == _observed_board()
    assert pipe._tsumo_count_2p == Counter({COLOR_RED: 20})
    assert pipe._constraint_valid_2p is False and pipe._chain_verify_pending_2p is None
    assert pipe._last_chain_event_for_settle_2p is None


def test_pending_candidate_is_never_committed_or_published() -> None:
    rec, _generations, _stream = _recorder()
    _pipe, ctx = _run_c6(rec)
    candidate = rec.active_candidates["2P"]["candidate"]
    assert not diag._grid_equal(ctx.confirmed_board, candidate.final_grid)
    receipt = rec.pending_receipt()
    assert receipt["commit_permission_issued"] is False
    assert receipt["completion_or_release_policy_connected"] is False


@dataclass(frozen=True)
class _SideResult:
    state: str
    confirmed_board: Any
    prob_board: Any
    board_none_reason: str | None = None


@dataclass(frozen=True)
class _PipelineResult:
    p1: _SideResult
    p2: _SideResult


def test_returned_confirmed_and_prob_are_really_isolated() -> None:
    rec, _generations, stream = _recorder()
    _pipe, ctx = _run_c6(rec)
    probabilistic = SimpleNamespace(to_board=lambda: ctx.confirmed_board.copy())
    side = _SideResult("stable", ctx.confirmed_board, probabilistic)
    result = rec.isolate_pipeline_result(_PipelineResult(side, side))
    assert result.p1.confirmed_board is not None
    assert result.p2.confirmed_board is None and result.p2.prob_board is None
    assert result.p2.board_none_reason == "pending_candidate_not_verified"
    row = [value for value in _rows(stream)
           if value.get("kind") == "c6_pending_publication"][-1]
    assert row["confirmed_equals_candidate"] is False
    assert row["prob_equals_candidate"] is False
    assert row["returned_confirmed"] is None and row["returned_prob"] is None
    assert row["state_machine_observation_preserved"] is True


def test_frame_side_outer_instrument_observes_isolated_return() -> None:
    rec, _generations, _stream = _recorder()
    _run_c6(rec)
    board = _observed_board()
    side = _SideResult("stable", board, None)
    seen: list[Any] = []

    class Pipeline:
        def update(self) -> _PipelineResult:
            return _PipelineResult(side, side)

    with contextlib.ExitStack() as stack:
        diag.install_pending_publication(stack, Pipeline, rec)
        inner = Pipeline.update

        def outer(pipe: Any) -> Any:
            result = inner(pipe)
            seen.append(result.p2.confirmed_board)
            return result

        diag.base.patch(stack, Pipeline, "update", outer)
        returned = Pipeline().update()
    assert seen == [None] and returned.p2.confirmed_board is None
    assert returned.p1.confirmed_board == board


def test_pending_side_discards_old_verifier_without_correction() -> None:
    rec, _generations, stream = _recorder()
    pipe, _ctx = _run_c6(rec)
    pipe._chain_verify_pending_2p = {"expected": _chain_board(), "cnn_history": [1, 2]}
    answer, correction = rec.isolate_old_verifier(pipe, "2P")
    assert answer == "pending_candidate_not_verified" and correction is None
    assert pipe._chain_verify_pending_2p is None
    row = [value for value in _rows(stream)
           if value.get("kind") == "c6_pending_old_verifier_isolated"][-1]
    assert row["residual_present"] is True and row["correction_applied"] is False


@pytest.mark.parametrize("side", ["1P", "2P"])
def test_candidate_contract_is_side_symmetric(side: str) -> None:
    rec, _generations, _stream = _recorder(side)
    pipe, _ctx = _run_c6(rec, side)
    assert rec.active_candidates[side]["candidate"].identity.side == side
    assert getattr(pipe, f"_constraint_valid_{side.lower()}") is False


def test_same_instance_c6_reuses_transaction_without_reconstruction() -> None:
    rec, _generations, stream = _recorder()
    _run_c6(rec)
    first = rec.active_candidates["2P"]["transaction"]
    _run_c6(rec)
    assert rec.active_candidates["2P"]["transaction"] is first
    assert len(rec.candidate_records) == 1
    assert any("candidate_reused_without_reconstruction" in str(row)
               for row in _rows(stream))


def test_rejected_prediction_never_reuses_old_revision_for_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec, generations, stream = _recorder()

    def episode_only(side: str, event: Any, _board: Any, _result: Any,
                     _caller: str) -> None:
        handle = rec.active_handles[side]
        rec.ledger.add_episode(
            handle, generation=rec._current_generation(side),
            frame_idx=FRAME, time_sec=TIME, event=event, capture_source="negative",
        )

    monkeypatch.setattr(rec, "record_c6_prediction", episode_only)
    _run_c6(rec)
    assert rec.candidate_records == [] and generations.generation("2P").action_revision == 25
    assert any(row.get("reason") == "c6_prediction_not_bound_to_actual_call"
               for row in _rows(stream))


def test_new_software_action_discards_pending_once() -> None:
    rec, generations, _stream = _recorder()
    _run_c6(rec)
    generations.set("2P", 26)
    rec.record_generation({"kind": "software_generation", "side": "2P",
        "reset_epoch": 0, "action_revision": 26, "frame_idx": FRAME,
        "time_sec": TIME, "clock_source": "pipeline_update_input",
        "reason": "state_machine_tsumo_fall_entry",
        "identity_scope": "software_observation_only_not_physical_identity",
        "commit_permission_issued": False})
    record = rec.candidate_records[0]
    assert record["transaction"].state.value == "discarded"
    assert "2P" not in rec.active_candidates


def test_cold_nonlanding_has_no_candidate_and_keeps_reason() -> None:
    rec, _generations, stream = _recorder(open_landing=False)
    pipe, _ctx = _run_c6(rec)
    assert rec.candidate_records == []
    assert pipe._last_chain_event_for_settle_2p is None
    assert any("landing_handle_or_physical_final_missing" in str(row)
               for row in _rows(stream))


class BoardState(Enum):
    CHAIN = "chain"
    GRAVITY_SETTLE = "gravity_settle"
    STABLE = "stable"


class _TransformPipeline:
    def _step_side(self, side: str, frame_idx: int, time_sec: float,
                   prev_state: Any, ctx: Any, _effective_chain_event: Any) -> Any:
        if (
            prev_state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE)
            and ctx.state == BoardState.STABLE
            and _effective_chain_event is not None
        ):
            if side == "1P":
                self._last_chain_event_for_settle_1p = None
            else:
                self._last_chain_event_for_settle_2p = None
            cr = self._chain_sim.simulate(_effective_chain_event.before_board)
            if cr.chain_count > 0 and cr.final_board is not None:
                final = cr.final_board.copy()
                ctx.confirmed_board = final
                ctx.pending_board = final.copy()
                self._chain_verify_pending_1p = {"expected": final}
                target_tsumo = self._tsumo_count_1p
                for color in target_tsumo:
                    target_tsumo[color] -= 1
                self._constraint_valid_1p = True
        return _transform_passthrough(ctx.confirmed_board)


def test_ast_transform_replaces_exact_c6_block_and_restores() -> None:
    calls: list[tuple[Any, ...]] = []
    original = _TransformPipeline._step_side
    transformed, receipt = diag._build_transformed_step(
        original, lambda *args: calls.append(args),
    )
    pipe = SimpleNamespace()
    ctx = SimpleNamespace(state=BoardState.STABLE, confirmed_board=_observed_board())
    result = transformed(pipe, "2P", FRAME, TIME, BoardState.GRAVITY_SETTLE,
                         ctx, SimpleNamespace())
    assert result == _observed_board() and len(calls) == 1
    assert receipt["matched_blocks"] == 1 and receipt["state_machine_rollback"] is False
    assert _TransformPipeline._step_side is original


def test_transform_captures_existing_shadow_global_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setitem(_TransformPipeline._step_side.__globals__,
                        "_transform_passthrough", lambda _value: sentinel)
    transformed, _receipt = diag._build_transformed_step(
        _TransformPipeline._step_side, lambda *_args: None,
    )
    ctx = SimpleNamespace(state=BoardState.STABLE, confirmed_board=_observed_board())
    result = transformed(SimpleNamespace(), "2P", FRAME, TIME,
                         BoardState.STABLE, ctx, None)
    assert result is sentinel


def test_ast_transform_rejects_missing_pattern() -> None:
    class Missing:
        def _step_side(self) -> None:
            return None

    with pytest.raises(RuntimeError, match="1件必要"):
        diag._build_transformed_step(Missing._step_side, lambda *_args: None)


def test_no_event_control_does_not_call_pending_hook() -> None:
    calls: list[Any] = []
    transformed, _receipt = diag._build_transformed_step(
        _TransformPipeline._step_side, lambda *args: calls.append(args),
    )
    pipe = SimpleNamespace()
    ctx = SimpleNamespace(state=BoardState.STABLE, confirmed_board=_observed_board())
    transformed(pipe, "2P", FRAME, TIME, BoardState.GRAVITY_SETTLE, ctx, None)
    assert calls == [] and ctx.confirmed_board == _observed_board()


def test_difference_receipt_counts_every_kind_and_intentional_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "ref"
    reference.mkdir()
    (reference / "frames.jsonl").write_text(
        '{"kind":"frame_side","x":1}\n{"kind":"raw","x":2}\n', encoding="utf-8")
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_text(
        '{"kind":"frame_side","x":9}\n{"kind":"raw","x":2}\n'
        '{"kind":"c6_pending_block","x":3}\n', encoding="utf-8")
    monkeypatch.setattr(diag, "REFERENCE", reference)
    result = diag._difference_receipt(candidate)
    assert result["changed_count"] == 1
    assert result["missing_keys"] == [] and result["added_keys"] == []
    assert result["actual_count"] == 2


def test_prefix_comparison_uses_time_window_not_row_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "ref"
    reference.mkdir()
    (reference / "frames.jsonl").write_text(
        '{"kind":"collector_snapshot","frame_idx":34798,"time_sec":579.966}\n'
        '{"kind":"collector_snapshot","frame_idx":34900,"time_sec":581.666}\n',
        encoding="utf-8")
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_text(
        '{"kind":"collector_snapshot","frame_idx":34798,"time_sec":579.966}\n'
        '{"kind":"collector_snapshot","frame_idx":35400,"time_sec":590.0}\n',
        encoding="utf-8")
    monkeypatch.setattr(diag.completion, "ATOMIC_REFERENCE", reference)
    result = diag._intentional_prefix_comparison(candidate)["kinds"]["collector_snapshot"]
    assert len(result["missing_keys"]) == 1 and result["added_keys"] == []


def test_window_stats_counts_only_requested_collector_side() -> None:
    rows = {
        "a": {"kind": "frame_side", "side": "1P", "frame_idx": 34148,
              "state": "stable"},
        "b": {"kind": "collector_snapshot", "side": "1P", "frame_idx": 34148},
        "c": {"kind": "collector_snapshot", "side": "2P", "frame_idx": 34258},
    }
    result = diag._window_stats(rows, "1P", 34080, 34380)
    assert result["frame_side_count"] == 1 and result["stable_count"] == 1
    assert result["collector_count"] == 1 and result["collector_frames"] == [34148]


def test_coverage_rejects_shifted_complete_sized_frame_set() -> None:
    rows = {}
    for frame in range(33602, 36302, 2):
        for side in ("1P", "2P"):
            for kind in ("frame_side", "raw_score_ocr", "completion_frame_observation"):
                rows[f"{kind}/{frame}/{side}"] = {
                    "kind": kind, "frame_idx": frame, "side": side,
                }
    result = diag._coverage(rows)
    assert all(not value["complete"] for value in result.values())
    assert all(value["missing_frame_side"] and value["extra_frame_side"]
               for value in result.values())


def test_prepare_records_no_rollback_and_all_new_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag, "ORIGINAL_PREPARE", lambda _args: (
        {"input_and_code_sha256": {}}, {},
    ))
    monkeypatch.setattr(diag.completion.prechain, "_reference_receipt",
                        lambda _root: ({"status": "complete"}, {"ref": "a" * 64}))
    receipt, _config = diag.pending_prepare(
        argparse.Namespace(start_sec=560.0, end_sec=605.0),
    )
    contract = receipt["c6_pending_commit_shadow"]
    assert contract["state_machine_rollback"] is False
    assert contract["release_policy_connected"] is False
    assert receipt["input_and_code_sha256"][str(diag.TEST)]


def test_run_restores_prediction_extension_points_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = ("ledger_prepare", "PredictionLedgerRecorder", "ledger_instrument", "ledger_finish")
    originals = tuple(getattr(diag.prediction, name) for name in names)
    monkeypatch.setattr(diag.prediction, "run",
                        lambda _args: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        diag.run(argparse.Namespace(output_root=tmp_path))
    assert tuple(getattr(diag.prediction, name) for name in names) == originals


def test_source_has_no_top_level_src_import_and_functions_are_short() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any(getattr(node, "module", "") == "src" for node in imports)
    too_long = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = int(node.end_lineno or node.lineno) - node.lineno + 1
            if length > 50:
                too_long.append((node.name, length))
    assert too_long == []
