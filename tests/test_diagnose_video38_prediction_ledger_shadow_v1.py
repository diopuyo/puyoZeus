"""prediction ledger shadow adapterの観測専用・非変更CPU検査。"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import diagnose_video38_prediction_ledger_shadow_v1 as diag
from src.board import Board, COLOR_BLUE, COLOR_RED
from src.chain import ChainSimulator
from src.chain_detector import CHAIN_MECHANISM_LANDING, ChainEvent
from src import chain_prediction_ledger_v1 as ledger_module
from src.scoring import calculate_chain_score


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/diagnose_video38_prediction_ledger_shadow_v1.py"
FRAME, TIME = 34702, 578.3666666666667


class _GenerationRecorder:
    """adapter単体用。software世代の物理的正しさを模擬しない。"""

    def __init__(self, action: int | None = 11) -> None:
        self.values = {
            side: SimpleNamespace(side=side, reset_epoch=3, action_revision=action)
            for side in ("1P", "2P")
        }
        self._in_frame = False
        self._frame: int | None = None
        self._time: float | None = None

    def generation(self, side: str) -> Any:
        return self.values[side]

    def set(self, side: str, reset: int, action: int | None) -> None:
        self.values[side] = SimpleNamespace(
            side=side, reset_epoch=reset, action_revision=action,
        )

    def begin(self, frame: int, time_sec: float) -> None:
        self._in_frame, self._frame, self._time = True, frame, time_sec

    def end(self) -> None:
        self._in_frame = False


class BoardStateMachine:
    """ledger_instrumentのclass取得だけに使うfixture。"""


class _Pipeline:
    """ledger_instrumentのclass取得だけに使うfixture。"""


def _board(extra: bool = False) -> Board:
    grid = [[0] * 6 for _ in range(13)]
    grid[12][0:4] = [COLOR_RED] * 4
    if extra:
        grid[11][5] = COLOR_BLUE
    return Board.from_list(grid)


def _result(board: Board) -> Any:
    return ChainSimulator().simulate(board)


def _event(board: Board, mechanism: str = CHAIN_MECHANISM_LANDING) -> ChainEvent:
    result = _result(board)
    score = calculate_chain_score(result).total_score
    return ChainEvent(
        trigger_sec=TIME, end_sec=TIME + 1.0, before_board=board,
        chain_count=max(1, result.chain_count), total_erased=result.total_erased,
        total_score=score, base_score=score, all_clear_bonus_applied=0,
        ojama_sent=0, leftover_score=0, is_all_clear=False,
        mechanism=mechanism, score_estimated=False,
    )


def _recorder(action: int | None = 11) -> tuple[Any, Any, io.StringIO]:
    stream = io.StringIO()
    rec = diag.PredictionLedgerRecorder(stream, {})
    generations = _GenerationRecorder(action)
    rec.bind_runtime(ledger_module, generations)
    _begin(rec, generations, FRAME, TIME)
    return rec, generations, stream


def _begin(rec: Any, generations: _GenerationRecorder,
           frame: int, time_sec: float) -> None:
    rec.begin_frame(frame, time_sec)
    generations.begin(frame, time_sec)


def _open(rec: Any, board: Board | None = None) -> Any:
    value = _board() if board is None else board
    rec.record_start("2P", _event(value), _result(value))
    return rec.all_handles[-1]


def _rows(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_landing_creates_one_provisional_and_origin_prediction() -> None:
    rec, _generations, _stream = _recorder()
    handle = _open(rec)
    snap = rec.ledger.snapshot(handle)
    assert len(snap.episodes) == 1 and len(snap.predictions) == 1
    assert snap.origin_prediction_revision == 1
    assert snap.predictions[0].is_origin_reference is True


def test_start_recapture_adds_episode_without_new_handle() -> None:
    rec, generations, _stream = _recorder()
    handle = _open(rec)
    _begin(rec, generations, FRAME + 54, TIME + 0.9)
    board = _board()
    rec.record_start("2P", _event(board, "formula_read"), _result(board))
    snap = rec.ledger.snapshot(handle)
    assert rec.all_handles == [handle]
    assert len(snap.episodes) == 2 and len(snap.predictions) == 2
    assert snap.origin_prediction_revision == 1


def test_existing_completion_kinds_and_values_are_still_emitted() -> None:
    rec, _generations, stream = _recorder()
    _open(rec)
    rows = _rows(stream)
    legacy = [row for row in rows if not diag._is_new_kind(row.get("kind"))]
    assert [row["kind"] for row in legacy] == [
        "completion_lineage_start", "physical_prediction",
    ]
    assert legacy[1]["prediction"]["chain_count"] == 1


def test_cold_start_nonlanding_keeps_old_rows_and_records_reason() -> None:
    rec, _generations, stream = _recorder()
    board = _board()
    rec.record_start("2P", _event(board, "formula_read"), _result(board))
    assert rec.all_handles == []
    rows = _rows(stream)
    assert any(row.get("kind") == "completion_lineage_start" for row in rows)
    assert any("first_event_not_landing" in str(row) for row in rows)
    assert any("no_landing_provisional" in str(row) for row in rows)


def test_unknown_action_is_not_filled_when_landing_opens() -> None:
    rec, _generations, _stream = _recorder(action=None)
    handle = _open(rec)
    assert rec.ledger.snapshot(handle).generation.action_revision is None


def test_known_generation_transition_invalidates_at_its_real_clock() -> None:
    rec, generations, _stream = _recorder()
    handle = _open(rec)
    _begin(rec, generations, FRAME + 2, TIME + 0.04)
    generations.set("2P", 3, 12)
    rec.record_generation({
        "kind": "software_generation", "side": "2P", "reset_epoch": 3,
        "action_revision": 12, "frame_idx": FRAME + 2, "time_sec": TIME + 0.04,
        "clock_source": "pipeline_update_input", "reason": "tsumo_fall",
        "identity_scope": "software_observation_only_not_physical_identity",
        "commit_permission_issued": False,
    })
    snap = rec.ledger.snapshot(handle)
    assert snap.status.value == "invalidated"
    assert snap.invalidation.invalidated_at.frame_idx == FRAME + 2
    assert "2P" not in rec.active_handles


def test_outside_update_generation_clock_stays_none_until_next_observation() -> None:
    rec, generations, stream = _recorder()
    handle = _open(rec)
    generations.end()
    generations.set("2P", 4, None)
    rec.record_generation({
        "kind": "software_generation", "side": "2P", "reset_epoch": 4,
        "action_revision": None, "frame_idx": None, "time_sec": None,
        "clock_source": "outside_update_time_unknown", "reason": "reset",
        "identity_scope": "software_observation_only_not_physical_identity",
        "commit_permission_issued": False,
    })
    assert rec.ledger.snapshot(handle).status.value == "provisional"
    owner = SimpleNamespace(step_count=0, total_power=0, last_valid_t=TIME)
    rec.record_formula_reset("2P", owner, "reset")
    assert rec.ledger.snapshot(handle).status.value == "provisional"
    outside = [r for r in _rows(stream)
               if r.get("kind") == "prediction_ledger_rejected"][-1]
    assert outside["frame_idx"] is None and outside["time_sec"] is None
    assert outside["clock_source"] == "outside_update_time_unknown"
    _begin(rec, generations, FRAME + 4, TIME + 0.08)
    delta = SimpleNamespace(
        side="2P", prev_score=971, cur_score=971, delta=0, is_valid=True,
    )
    rec.record_score(delta, 971)
    snap = rec.ledger.snapshot(handle)
    assert snap.invalidation.invalidated_at.frame_idx == FRAME + 4
    generation_rows = [r for r in _rows(stream) if r.get("kind") == "software_generation"]
    assert generation_rows[-1]["frame_idx"] is None
    invalidation = next(r for r in _rows(stream)
                        if r.get("kind") == "prediction_ledger_generation_invalidation")
    assert invalidation["value"]["transition_observation"]["frame_idx"] is None


def test_raw_score_is_saved_only_as_evidence_without_anchor() -> None:
    rec, generations, _stream = _recorder()
    handle = _open(rec)
    _begin(rec, generations, FRAME + 2, TIME + 0.04)
    delta = SimpleNamespace(
        side="2P", prev_score=971, cur_score=1011, delta=40, is_valid=True,
    )
    rec.record_score(delta, 1011)
    snap = rec.ledger.snapshot(handle)
    assert snap.raw_score_evidence[-1].delta == 40
    assert not hasattr(snap, "entry_score_anchor")


def test_initial_formula_reset_binds_step_zero_session() -> None:
    rec, generations, _stream = _recorder()
    handle = _open(rec)
    _begin(rec, generations, 34718, 578.6333333333)
    owner = SimpleNamespace(step_count=0, total_power=0, last_valid_t=578.6333333333)
    rec.record_formula_reset("2P", owner, "update")
    snap = rec.ledger.snapshot(handle)
    assert snap.formula_session_binding.session_id == 1
    assert snap.formula_evidence[-1].step_index == 0


def test_formula_session_change_invalidates_old_handle() -> None:
    rec, generations, _stream = _recorder()
    handle = _open(rec)
    owner = SimpleNamespace(step_count=0, total_power=0, last_valid_t=578.6)
    _begin(rec, generations, 34718, 578.6333333333)
    rec.record_formula_reset("2P", owner, "update")
    _begin(rec, generations, 34720, 578.6666666667)
    rec.record_formula_reset("2P", owner, "update")
    assert rec.ledger.snapshot(handle).status.value == "invalidated"
    assert any("formula session" in row["reason"] for row in rec.rejections)


def test_prediction_with_unmatched_input_keeps_old_row_and_rejection() -> None:
    rec, generations, stream = _recorder()
    _open(rec)
    _begin(rec, generations, FRAME + 2, TIME + 0.04)
    rec.record_prediction("2P", _result(_board(extra=True)), "_step_side/C-6")
    rows = _rows(stream)
    assert sum(r.get("kind") == "physical_prediction" for r in rows) == 2
    assert any("prediction_without_exact_start_episode" in str(r) for r in rows)


def test_chain_zero_prediction_is_not_filled_or_bound() -> None:
    rec, _generations, stream = _recorder()
    board = Board()
    rec.record_start("2P", _event(board), _result(board))
    rows = _rows(stream)
    prediction = next(r for r in rows if r.get("kind") == "physical_prediction")
    assert prediction["prediction"]["chain_count"] == 0
    assert any("chain_result_empty_or_unknown" in str(r) for r in rows)
    assert len(rec.ledger.snapshot(rec.all_handles[0]).predictions) == 0


def test_c6_wrapper_binds_actual_event_and_input_without_second_simulation() -> None:
    rec, _generations, stream = _recorder()
    handle = _open(rec)
    board = _board(extra=True)
    event = _event(board, "formula_read")

    class Simulator:
        calls = 0

        def simulate(self, input_board: Board) -> Any:
            self.calls += 1
            return _result(input_board)

    simulator = Simulator()
    with contextlib.ExitStack() as stack:
        diag.base.patch(
            stack, diag.completion, "CompletionRecorder", diag.PredictionLedgerRecorder,
        )
        diag.instrument_simulate_exact(stack, Simulator, rec)

        def _step_side() -> Any:
            side = "2P"
            _effective_chain_event = event
            assert side and _effective_chain_event
            return simulator.simulate(board)

        result = _step_side()
    snap = rec.ledger.snapshot(handle)
    assert simulator.calls == 1 and result.chain_count == 1
    assert len(snap.episodes) == 2 and len(snap.predictions) == 2
    assert snap.predictions[-1].episode_revision == snap.episodes[-1].episode_revision
    assert sum(r.get("kind") == "physical_prediction" for r in _rows(stream)) == 2


def test_c6_wrapper_missing_effective_event_keeps_legacy_and_rejects() -> None:
    rec, _generations, stream = _recorder()
    _open(rec)
    board = _board(extra=True)

    class Simulator:
        def simulate(self, input_board: Board) -> Any:
            return _result(input_board)

    simulator = Simulator()
    with contextlib.ExitStack() as stack:
        diag.instrument_simulate_exact(stack, Simulator, rec)

        def _step_side() -> Any:
            side = "2P"
            _effective_chain_event = None
            assert side and _effective_chain_event is None
            return simulator.simulate(board)

        _step_side()
    assert sum(r.get("kind") == "physical_prediction" for r in _rows(stream)) == 2
    assert any("effective_event_unknown" in str(r) for r in _rows(stream))


def test_ledger_receipt_omits_owner_token_and_all_permissions() -> None:
    rec, _generations, _stream = _recorder()
    _open(rec)
    value = rec.ledger_receipt()
    payload = json.dumps(value, ensure_ascii=False)
    assert "_owner_token" not in payload
    assert value["commit_permission_issued"] is False
    assert value["anchor_binding_performed"] is False
    assert value["completion_or_publication_evaluated"] is False


def test_full_legacy_comparison_ignores_only_new_namespaced_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "reference"
    reference.mkdir()
    expected = [{"kind": "frame_side", "x": 1}, {"kind": "physical_prediction", "x": 2}]
    (reference / "frames.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in expected), encoding="utf-8",
    )
    candidate = tmp_path / "candidate.jsonl"
    rows = [expected[0], {"kind": "software_generation", "x": 9},
            {"kind": "prediction_ledger_episode", "x": 8}, expected[1]]
    candidate.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(diag, "REFERENCE", reference)
    result = diag._full_legacy_comparison(candidate)
    assert result["row_count"] == 2 and result["all_legacy_rows_bit_exact"] is True


def test_full_legacy_comparison_rejects_changed_old_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "frames.jsonl").write_text('{"kind":"frame_side","x":1}\n', encoding="utf-8")
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_text('{"kind":"frame_side","x":2}\n', encoding="utf-8")
    monkeypatch.setattr(diag, "REFERENCE", reference)
    with pytest.raises(RuntimeError, match="全行不一致"):
        diag._full_legacy_comparison(candidate)


def test_dynamic_current_ledger_load_restores_existing_modules() -> None:
    names = ("src.chain_commit_candidate_v1", "src.chain_prediction_ledger_v1")
    previous = {name: sys.modules.pop(name, None) for name in names}
    guards = {str(diag.COMMIT_SOURCE): diag.base.sha256(diag.COMMIT_SOURCE),
              str(diag.LEDGER_SOURCE): diag.base.sha256(diag.LEDGER_SOURCE)}
    try:
        with contextlib.ExitStack() as stack:
            loaded = diag._load_current_ledger(stack, guards)
            assert sys.modules[names[1]] is loaded
            assert Path(str(loaded.__file__)).resolve() == diag.LEDGER_SOURCE.resolve()
            assert loaded.ChainPredictionLedger is not None
        assert all(name not in sys.modules for name in names)
    finally:
        sys.modules.update({name: value for name, value in previous.items()
                            if value is not None})


def test_dynamic_load_rejects_unexpected_existing_module() -> None:
    name = "src.chain_prediction_ledger_v1"
    assert name in sys.modules
    with contextlib.ExitStack() as stack, pytest.raises(RuntimeError, match="既load"):
        diag._load_guarded(
            stack, name, diag.LEDGER_SOURCE, diag.base.sha256(diag.LEDGER_SOURCE),
        )


def test_dynamic_load_rejects_wrong_guard_before_execution() -> None:
    with contextlib.ExitStack() as stack, pytest.raises(ValueError, match="SHA不一致"):
        diag._load_guarded(
            stack, "src.chain_prediction_ledger_v1", diag.LEDGER_SOURCE, "0" * 64,
        )


def test_prepare_adds_current_and_full_reference_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag, "ORIGINAL_PREPARE", lambda _args: ({
        "input_and_code_sha256": {},
    }, {"config": True}))
    monkeypatch.setattr(
        diag.completion.prechain, "_reference_receipt",
        lambda _root: ({"status": "complete"}, {"reference-file": "a" * 64}),
    )
    args = argparse.Namespace(start_sec=560.0, end_sec=605.0)
    receipt, config = diag.ledger_prepare(args)
    assert config == {"config": True}
    assert receipt["input_and_code_sha256"][str(diag.LEDGER_SOURCE)]
    assert receipt["input_and_code_sha256"][str(diag.COMMIT_TEST)]
    assert receipt["input_and_code_sha256"]["reference-file"] == "a" * 64
    assert diag._PREPARED_GUARDS == receipt["input_and_code_sha256"]
    diag._PREPARED_GUARDS = None


def test_instrument_installs_original_before_generation_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    rec = diag.PredictionLedgerRecorder(io.StringIO(), {})
    collector = SimpleNamespace(RecognitionPipeline=_Pipeline)
    monkeypatch.setattr(diag, "_PREPARED_GUARDS", {})
    def original(_stack: Any, _collector: Any, _rec: Any) -> None:
        assert diag.completion.instrument_simulate is diag.instrument_simulate_exact
        order.append("original")

    monkeypatch.setattr(diag, "ORIGINAL_INSTRUMENT", original)
    monkeypatch.setattr(diag, "_load_current_ledger",
                        lambda _stack, _guards: ledger_module)
    monkeypatch.setattr(diag.generation_hooks, "install_generation_hooks",
                        lambda *_args: order.append("generation"))
    with contextlib.ExitStack() as stack:
        diag.ledger_instrument(stack, collector, rec)
    assert order == ["original", "generation"]
    assert rec.ledger is not None and rec.generation_recorder is not None
    assert diag.completion.instrument_simulate is not diag.instrument_simulate_exact


def test_run_restores_completion_patch_points_after_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion = diag.completion
    originals = (completion.completion_prepare, completion.CompletionRecorder,
                 completion.completion_instrument, completion.completion_finish)

    def fail(_args: Any) -> Any:
        assert completion.completion_prepare is diag.ledger_prepare
        assert completion.CompletionRecorder is diag.PredictionLedgerRecorder
        raise RuntimeError("fixture-stop")

    monkeypatch.setattr(completion, "run", fail)
    with pytest.raises(RuntimeError, match="fixture-stop"):
        diag.run(argparse.Namespace(output_root=tmp_path, start_sec=560.0, end_sec=605.0))
    assert (completion.completion_prepare, completion.CompletionRecorder,
            completion.completion_instrument, completion.completion_finish) == originals
    assert diag._PREPARED_GUARDS is None


def test_finish_adds_ledger_receipt_to_complete_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec, _generations, _stream = _recorder()
    _open(rec)
    (tmp_path / "frames.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(diag, "_full_legacy_comparison",
                        lambda _path: {"all_legacy_rows_bit_exact": True})

    def finish(output: Path, _receipt: Any, _rec: Any, _elapsed: float) -> dict[str, Any]:
        diag.base.write_json(output / "COMPLETE", {
            "status": "fixture", "sha256": {"COMPLETION_RECEIPT.json": "x"},
        })
        return {"status": "fixture"}

    monkeypatch.setattr(diag, "ORIGINAL_FINISH", finish)
    diag.ledger_finish(tmp_path, {"input_and_code_sha256": {}}, rec, 1.0)
    complete = json.loads((tmp_path / "COMPLETE").read_text(encoding="utf-8"))
    assert "LEDGER_RECEIPT.json" in complete["sha256"]
    assert (tmp_path / "LEDGER_RECEIPT.json").is_file()


def test_adapter_functions_respect_fifty_line_rule() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    too_long = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = int(node.end_lineno or node.lineno) - node.lineno + 1
            if length > 50:
                too_long.append((node.name, length))
    assert too_long == []
