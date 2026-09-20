"""chain instance / episode / prediction分離台帳のCPU契約テスト。"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.board import Board, COLOR_BLUE, COLOR_RED
from src.chain import ChainResult, ChainSimulator
from src.chain_detector import CHAIN_MECHANISM_LANDING, ChainEvent
from src.chain_prediction_ledger_v1 import (
    ActiveChainExistsError,
    ChainGeneration,
    ChainLedgerStatus,
    ChainPredictionLedger,
    LedgerValidationError,
    ObservationOrderError,
    PredictionScope,
    ProvisionalChainHandle,
    StaleChainHandleError,
)
from src.scoring import calculate_chain_score


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/chain_prediction_ledger_v1.py"
FRAME, TIME = 34702, 578.3666666666667


def _board(extra: bool = False) -> Board:
    """1連鎖を持つ小盤面を作る。"""
    grid = [[0] * 6 for _ in range(13)]
    grid[12][0:4] = [COLOR_RED] * 4
    if extra:
        grid[11][5] = COLOR_BLUE
    return Board.from_list(grid)


def _two_chain_board() -> Board:
    """青消去後に赤が落ちて2連鎖になる盤面を作る。"""
    grid = [[0] * 6 for _ in range(13)]
    grid[11][0:4] = [COLOR_BLUE] * 4
    grid[8][0] = COLOR_RED
    grid[9][0] = COLOR_RED
    grid[10][0] = COLOR_RED
    grid[12][0] = COLOR_RED
    return Board.from_list(grid)


def _result(board: Board | None = None) -> ChainResult:
    """通常ルールの物理予測を返す。"""
    return ChainSimulator().simulate(_board() if board is None else board)


def _first_step_only(result: ChainResult) -> ChainResult:
    """同じoriginから利用可能になった低い初期予測を模擬する。"""
    step = result.steps[0]
    return ChainResult(
        steps=[step], chain_count=1, total_erased=step.erased_count,
        total_ojama=step.erased_ojama, final_board=step.board_after,
        participating_cells=step.erased_count,
    )


def _event(board: Board, mechanism: str = CHAIN_MECHANISM_LANDING) -> ChainEvent:
    """盤面からCPU fixture用eventを作る。"""
    result = _result(board)
    score = calculate_chain_score(result).total_score
    return ChainEvent(
        trigger_sec=TIME, end_sec=TIME + 1.0, before_board=board,
        chain_count=max(1, result.chain_count), total_erased=result.total_erased,
        total_score=score, base_score=score, all_clear_bonus_applied=0,
        ojama_sent=0, leftover_score=0, is_all_clear=False,
        mechanism=mechanism, score_estimated=False,
    )


def _opened(
    *, side: str = "2P", reset: int | None = 3, action: int | None = 11,
) -> tuple[ChainPredictionLedger, Any, ChainGeneration, Board]:
    """各テストに独立ledgerとlanding handleを返す。"""
    ledger = ChainPredictionLedger()
    generation = ChainGeneration(side, reset, action)
    board = _board()
    handle = ledger.open_landing_provisional(
        generation=generation, frame_idx=FRAME, time_sec=TIME,
        origin_before_board=board, landing_event=_event(board),
    )
    return ledger, handle, generation, board


def _add_origin_prediction(
    ledger: ChainPredictionLedger, handle: Any,
    generation: ChainGeneration, board: Board,
) -> Any:
    """最初のorigin predictionを追加する。"""
    return ledger.add_prediction(
        handle, generation=generation, frame_idx=FRAME, time_sec=TIME,
        episode_revision=1, input_board=board, result=_result(board),
    )


def test_open_stores_provisional_origin_and_initial_episode() -> None:
    ledger, handle, generation, board = _opened()
    snap = ledger.snapshot(handle)
    assert snap.status is ChainLedgerStatus.PROVISIONAL
    assert snap.generation == generation and snap.handle is handle
    assert snap.copy_origin_board().to_dict() == board.to_dict()
    assert len(snap.episodes) == 1
    assert snap.episodes[0].capture_source == "landing"
    assert snap.episodes[0].episode_revision == 1


def test_open_copies_mutable_board_and_event_board() -> None:
    ledger = ChainPredictionLedger()
    generation, board = ChainGeneration("2P", 3, 11), _board()
    event = _event(board)
    handle = ledger.open_landing_provisional(
        generation=generation, frame_idx=FRAME, time_sec=TIME,
        origin_before_board=board, landing_event=event,
    )
    expected = ledger.snapshot(handle).origin_before_grid
    board._grid.fill(0)
    event.before_board._grid.fill(0)
    assert ledger.snapshot(handle).origin_before_grid == expected
    assert ledger.snapshot(handle).episodes[0].before_grid == expected


def test_snapshot_collections_are_immutable_tuples() -> None:
    ledger, handle, _generation, _board_value = _opened()
    snap = ledger.snapshot(handle)
    assert isinstance(snap.episodes, tuple)
    assert isinstance(snap.predictions, tuple)
    assert isinstance(snap.raw_score_evidence, tuple)
    assert isinstance(snap.formula_evidence, tuple)


def test_start_recapture_adds_episode_without_replacing_instance() -> None:
    ledger, handle, generation, board = _opened()
    episode = ledger.add_episode(
        handle, generation=generation, frame_idx=34756, time_sec=579.2666666667,
        event=_event(board, "formula_read"), capture_source="formula_start",
    )
    snap = ledger.snapshot(handle)
    assert snap.handle is handle and len(snap.episodes) == 2
    assert episode.episode_revision == 2 and episode.mechanism == "formula_read"
    assert snap.origin_before_grid == snap.episodes[0].before_grid


def test_origin_prediction_scope_is_derived_and_frozen() -> None:
    ledger, handle, generation, board = _opened()
    prediction = _add_origin_prediction(ledger, handle, generation, board)
    snap = ledger.snapshot(handle)
    assert prediction.scope is PredictionScope.TOTAL_FROM_INSTANCE_ORIGIN
    assert prediction.is_origin_reference is True
    assert snap.origin_prediction_revision == prediction.prediction_revision == 1


def test_revision_board_prediction_is_remaining_and_never_origin() -> None:
    ledger, handle, generation, _board_value = _opened()
    revision_board = _board(extra=True)
    episode = ledger.add_episode(
        handle, generation=generation, frame_idx=FRAME + 2, time_sec=TIME + 0.04,
        event=_event(revision_board, "formula_read"), capture_source="mid_chain",
    )
    prediction = ledger.add_prediction(
        handle, generation=generation, frame_idx=FRAME + 2, time_sec=TIME + 0.04,
        episode_revision=episode.episode_revision,
        input_board=revision_board, result=_result(revision_board),
    )
    assert prediction.scope is PredictionScope.REMAINING_FROM_REVISION_BOARD
    assert prediction.is_origin_reference is False
    assert ledger.snapshot(handle).origin_prediction_revision is None


def test_later_larger_prediction_cannot_replace_origin_reference() -> None:
    ledger, generation, board = ChainPredictionLedger(), ChainGeneration("2P", 3, 11), _two_chain_board()
    handle = ledger.open_landing_provisional(
        generation=generation, frame_idx=FRAME, time_sec=TIME,
        origin_before_board=board, landing_event=_event(board),
    )
    full = _result(board)
    first = ledger.add_prediction(
        handle, generation=generation, frame_idx=FRAME, time_sec=TIME,
        episode_revision=1, input_board=board, result=_first_step_only(full),
    )
    larger = ledger.add_prediction(
        handle, generation=generation, frame_idx=FRAME + 2, time_sec=TIME + 0.04,
        episode_revision=1, input_board=board, result=full,
    )
    snap = ledger.snapshot(handle)
    assert larger.calculated_total_score > first.calculated_total_score
    assert larger.scope is PredictionScope.TOTAL_FROM_INSTANCE_ORIGIN
    assert larger.is_origin_reference is False
    assert snap.origin_prediction_revision == first.prediction_revision


def test_late_first_origin_prediction_is_not_promoted_retroactively() -> None:
    ledger, handle, generation, board = _opened()
    prediction = ledger.add_prediction(
        handle, generation=generation, frame_idx=FRAME + 2, time_sec=TIME + 0.04,
        episode_revision=1, input_board=board, result=_result(board),
    )
    assert prediction.scope is PredictionScope.TOTAL_FROM_INSTANCE_ORIGIN
    assert prediction.is_origin_reference is False
    assert ledger.snapshot(handle).origin_prediction_revision is None


def test_prediction_keeps_available_time_and_detached_grids() -> None:
    ledger, handle, generation, board = _opened()
    result = _result(board)
    prediction = ledger.add_prediction(
        handle, generation=generation, frame_idx=FRAME + 4, time_sec=TIME + 0.08,
        episode_revision=1, input_board=board, result=result,
    )
    input_saved, final_saved = prediction.input_grid, prediction.final_grid
    board._grid.fill(0)
    result.final_board._grid.fill(COLOR_BLUE)
    saved = ledger.snapshot(handle).predictions[0]
    assert saved.input_grid == input_saved and saved.final_grid == final_saved
    assert saved.available_at.frame_idx == FRAME + 4


def test_episode_and_prediction_keep_same_frame_insertion_order() -> None:
    ledger, handle, generation, board = _opened()
    episode = ledger.add_episode(
        handle, generation=generation, frame_idx=FRAME, time_sec=TIME,
        event=_event(board, "formula_read"), capture_source="same_frame_formula",
    )
    prediction = ledger.add_prediction(
        handle, generation=generation, frame_idx=FRAME, time_sec=TIME,
        episode_revision=episode.episode_revision, input_board=board, result=_result(board),
    )
    assert episode.ledger_sequence == 2
    assert prediction.ledger_sequence == 3


def test_initial_formula_reset_binds_session_without_invalidating_origin() -> None:
    ledger, handle, generation, board = _opened()
    _add_origin_prediction(ledger, handle, generation, board)
    evidence = ledger.record_formula(
        handle, generation=generation, frame_idx=34718, time_sec=578.6333333333,
        source="FormulaStepAccumulator._reset_session", session_id=2,
        step_index=0, total_power=0, step_product=None,
    )
    snap = ledger.snapshot(handle)
    assert snap.status is ChainLedgerStatus.PROVISIONAL
    assert snap.formula_session_binding is not None
    assert snap.formula_session_binding.session_id == 2
    assert evidence.step_index == 0 and snap.origin_prediction_revision == 1


def test_first_formula_step_uses_same_bound_session_and_real_time() -> None:
    ledger, handle, generation, _board_value = _opened()
    ledger.record_formula(
        handle, generation=generation, frame_idx=34718, time_sec=578.6333333333,
        source="reset", session_id=2, step_index=0, total_power=0, step_product=None,
    )
    step = ledger.record_formula(
        handle, generation=generation, frame_idx=34720, time_sec=578.6666666667,
        source="formula_observation", session_id=2,
        step_index=1, total_power=40, step_product=40,
    )
    assert step.step_index == 1 and step.total_power == step.step_product == 40
    assert ledger.snapshot(handle).formula_session_binding.observed_at.frame_idx == 34718


def test_step_two_uses_measured_frame_34804_not_start_recapture_frame() -> None:
    ledger, handle, generation, _board_value = _opened()
    ledger.record_formula(
        handle, generation=generation, frame_idx=34718, time_sec=578.6333333333,
        source="reset", session_id=2, step_index=0, total_power=0, step_product=None,
    )
    step = ledger.record_formula(
        handle, generation=generation, frame_idx=34804, time_sec=580.0666666667,
        source="formula_observation", session_id=2,
        step_index=2, total_power=700, step_product=660,
    )
    assert step.observed_at.frame_idx == 34804 and step.total_power == 700


def test_formula_session_change_is_rejected_without_overwrite() -> None:
    ledger, handle, generation, _board_value = _opened()
    first = ledger.record_formula(
        handle, generation=generation, frame_idx=34718, time_sec=578.6333333333,
        source="reset", session_id=2, step_index=0, total_power=0, step_product=None,
    )
    with pytest.raises(StaleChainHandleError):
        ledger.record_formula(
            handle, generation=generation, frame_idx=34720, time_sec=578.6666666667,
            source="new_session", session_id=3,
            step_index=1, total_power=40, step_product=40,
        )
    snap = ledger.snapshot(handle)
    assert snap.formula_session_binding.session_id == 2
    assert snap.formula_evidence == (first,)
    assert snap.status is ChainLedgerStatus.INVALIDATED
    assert snap.invalidation.reason == "formula_session_changed:2->3"
    with pytest.raises(StaleChainHandleError):
        ledger.record_raw_score(
            handle, generation=generation, frame_idx=34722, time_sec=578.7,
            source="old_session_after_change", raw_value=971,
            prev_score=971, cur_score=971, delta=0, is_valid=True,
        )


def test_raw_score_evidence_is_kept_without_anchor_or_permission() -> None:
    ledger, handle, generation, _board_value = _opened()
    raw = ledger.record_raw_score(
        handle, generation=generation, frame_idx=34702, time_sec=TIME,
        source="ScoreDelta/_apply_read", raw_value=971,
        prev_score=970, cur_score=971, delta=1, is_valid=True,
    )
    snap = ledger.snapshot(handle)
    assert raw.prev_score == 970 and raw.raw_value == 971
    assert snap.raw_score_evidence == (raw,)
    assert not hasattr(snap, "entry_score_anchor")
    assert not hasattr(ledger, "commit") and not hasattr(ledger, "release")


def test_raw_score_missing_is_preserved_as_none() -> None:
    ledger, handle, generation, _board_value = _opened()
    raw = ledger.record_raw_score(
        handle, generation=generation, frame_idx=34718, time_sec=578.6333333333,
        source="raw_score_ocr", raw_value=None, prev_score=None,
        cur_score=None, delta=None, is_valid=False,
    )
    assert raw.raw_value is None and raw.delta is None


@pytest.mark.parametrize("reset,action", [(None, None), (None, 4), (3, None)])
def test_unknown_generation_is_preserved(reset: int | None, action: int | None) -> None:
    ledger, handle, generation, _board_value = _opened(reset=reset, action=action)
    assert ledger.snapshot(handle).generation == generation


@pytest.mark.parametrize(
    "generation",
    [ChainGeneration("bad", 1, 1), ChainGeneration("1P", True, 1),
     ChainGeneration("1P", -1, 1), ChainGeneration("1P", 1, True),
     ChainGeneration("1P", 1, -1)],
)
def test_invalid_generation_is_rejected(generation: ChainGeneration) -> None:
    ledger, board = ChainPredictionLedger(), _board()
    with pytest.raises(LedgerValidationError):
        ledger.open_landing_provisional(
            generation=generation, frame_idx=FRAME, time_sec=TIME,
            origin_before_board=board, landing_event=_event(board),
        )


@pytest.mark.parametrize(
    "field,value",
    [("side", "1P"), ("reset_epoch", 4), ("action_revision", 12),
     ("reset_epoch", None), ("action_revision", None)],
)
def test_generation_mismatch_rejects_append(field: str, value: object) -> None:
    ledger, handle, generation, board = _opened()
    changed = replace(generation, **{field: value})
    with pytest.raises(StaleChainHandleError):
        ledger.add_episode(
            handle, generation=changed, frame_idx=FRAME + 2, time_sec=TIME + 0.04,
            event=_event(board, "formula_read"), capture_source="mismatch",
        )
    assert len(ledger.snapshot(handle).episodes) == 1


@pytest.mark.parametrize(
    "frame,time_value",
    [(FRAME - 1, TIME), (FRAME, TIME - 0.01), (FRAME - 1, TIME - 0.01)],
)
def test_frame_or_time_regression_is_rejected(frame: int, time_value: float) -> None:
    ledger, handle, generation, board = _opened()
    with pytest.raises(ObservationOrderError):
        ledger.add_episode(
            handle, generation=generation, frame_idx=frame, time_sec=time_value,
            event=_event(board, "formula_read"), capture_source="past",
        )


def test_duplicate_active_side_is_rejected_even_for_new_generation() -> None:
    ledger, _handle, _generation, _board_value = _opened()
    board = _board()
    with pytest.raises(ActiveChainExistsError):
        ledger.open_landing_provisional(
            generation=ChainGeneration("2P", 4, 12),
            frame_idx=FRAME + 2, time_sec=TIME + 0.04,
            origin_before_board=board, landing_event=_event(board),
        )


def test_invalidation_blocks_old_handle_and_same_generation_reopen() -> None:
    ledger, handle, generation, board = _opened()
    receipt = ledger.invalidate(
        handle, generation=generation, frame_idx=FRAME + 2,
        time_sec=TIME + 0.04, reason="new_action_observed",
    )
    assert ledger.snapshot(handle).status is ChainLedgerStatus.INVALIDATED
    assert receipt.reason == "new_action_observed"
    with pytest.raises(StaleChainHandleError):
        ledger.add_episode(
            handle, generation=generation, frame_idx=FRAME + 4, time_sec=TIME + 0.08,
            event=_event(board, "formula_read"), capture_source="resume",
        )
    with pytest.raises(StaleChainHandleError):
        ledger.open_landing_provisional(
            generation=generation, frame_idx=FRAME + 4, time_sec=TIME + 0.08,
            origin_before_board=board, landing_event=_event(board),
        )


def test_new_action_can_open_only_after_old_handle_is_invalidated() -> None:
    ledger, handle, generation, _board_value = _opened()
    ledger.invalidate(
        handle, generation=generation, frame_idx=FRAME + 2,
        time_sec=TIME + 0.04, reason="action_advanced",
    )
    board = _board(extra=True)
    next_generation = ChainGeneration("2P", 3, 12)
    next_handle = ledger.open_landing_provisional(
        generation=next_generation, frame_idx=FRAME + 4, time_sec=TIME + 0.08,
        origin_before_board=board, landing_event=_event(board),
    )
    assert next_handle.instance_id != handle.instance_id


@pytest.mark.parametrize(
    "next_generation",
    [ChainGeneration("2P", 2, 99), ChainGeneration("2P", 3, 10)],
)
def test_known_generation_cannot_reopen_at_smaller_value(
    next_generation: ChainGeneration,
) -> None:
    ledger, handle, generation, _board_value = _opened()
    ledger.invalidate(
        handle, generation=generation, frame_idx=FRAME + 2,
        time_sec=TIME + 0.04, reason="generation_changed",
    )
    board = _board(extra=True)
    with pytest.raises(StaleChainHandleError):
        ledger.open_landing_provisional(
            generation=next_generation, frame_idx=FRAME + 4, time_sec=TIME + 0.08,
            origin_before_board=board, landing_event=_event(board),
        )


def test_unknown_to_known_generation_creates_new_handle_not_upgrade() -> None:
    ledger, old, unknown, _board_value = _opened(reset=None, action=None)
    ledger.invalidate(
        old, generation=unknown, frame_idx=FRAME + 2,
        time_sec=TIME + 0.04, reason="external_generation_became_known",
    )
    board = _board(extra=True)
    known = ChainGeneration("2P", 4, 12)
    new = ledger.open_landing_provisional(
        generation=known, frame_idx=FRAME + 4, time_sec=TIME + 0.08,
        origin_before_board=board, landing_event=_event(board),
    )
    assert new.instance_id != old.instance_id
    assert ledger.snapshot(old).generation == unknown
    assert ledger.snapshot(new).generation == known


def test_other_side_has_independent_active_handle_and_clock() -> None:
    ledger, handle_2p, _generation, _board_value = _opened()
    board = _board()
    handle_1p = ledger.open_landing_provisional(
        generation=ChainGeneration("1P", 3, 7), frame_idx=FRAME - 100,
        time_sec=TIME - 2.0, origin_before_board=board, landing_event=_event(board),
    )
    assert ledger.snapshot(handle_1p).generation.side == "1P"
    assert ledger.snapshot(handle_2p).generation.side == "2P"


def test_reconstructed_or_foreign_handle_is_rejected() -> None:
    ledger, handle, generation, board = _opened()
    forged = ProvisionalChainHandle(handle.instance_id, handle.side, handle._owner_token)
    with pytest.raises(StaleChainHandleError):
        ledger.add_episode(
            forged, generation=generation, frame_idx=FRAME + 2, time_sec=TIME + 0.04,
            event=_event(board, "formula_read"), capture_source="forged",
        )
    other = ChainPredictionLedger()
    with pytest.raises(StaleChainHandleError):
        other.snapshot(handle)


def test_non_landing_event_cannot_create_origin() -> None:
    ledger, board = ChainPredictionLedger(), _board()
    with pytest.raises(LedgerValidationError):
        ledger.open_landing_provisional(
            generation=ChainGeneration("2P", 3, 11), frame_idx=FRAME, time_sec=TIME,
            origin_before_board=board, landing_event=_event(board, "formula_read"),
        )


def test_landing_event_before_board_must_equal_origin() -> None:
    ledger, origin, other = ChainPredictionLedger(), _board(), _board(extra=True)
    with pytest.raises(LedgerValidationError):
        ledger.open_landing_provisional(
            generation=ChainGeneration("2P", 3, 11), frame_idx=FRAME, time_sec=TIME,
            origin_before_board=origin, landing_event=_event(other),
        )


def test_prediction_must_reference_existing_episode() -> None:
    ledger, handle, generation, board = _opened()
    with pytest.raises(LedgerValidationError):
        ledger.add_prediction(
            handle, generation=generation, frame_idx=FRAME, time_sec=TIME,
            episode_revision=2, input_board=board, result=_result(board),
        )


def test_empty_chain_result_is_rejected_without_sequence_consumption() -> None:
    ledger, handle, generation, board = _opened()
    empty = _result(Board())
    with pytest.raises(LedgerValidationError):
        ledger.add_prediction(
            handle, generation=generation, frame_idx=FRAME, time_sec=TIME,
            episode_revision=1, input_board=board, result=empty,
        )
    episode = ledger.add_episode(
        handle, generation=generation, frame_idx=FRAME, time_sec=TIME,
        event=_event(board, "formula_read"), capture_source="after_failure",
    )
    assert episode.ledger_sequence == 2


def test_prediction_input_must_match_referenced_episode() -> None:
    ledger, handle, generation, _board_value = _opened()
    other = _board(extra=True)
    with pytest.raises(LedgerValidationError):
        ledger.add_prediction(
            handle, generation=generation, frame_idx=FRAME, time_sec=TIME,
            episode_revision=1, input_board=other, result=_result(other),
        )


def test_prediction_result_first_board_must_match_input() -> None:
    ledger, handle, generation, board = _opened()
    with pytest.raises(LedgerValidationError):
        ledger.add_prediction(
            handle, generation=generation, frame_idx=FRAME, time_sec=TIME,
            episode_revision=1, input_board=board, result=_result(_board(extra=True)),
        )


def test_prediction_last_step_must_match_final_board() -> None:
    ledger, handle, generation, board = _opened()
    broken = replace(_result(board), final_board=_board(extra=True))
    with pytest.raises(LedgerValidationError):
        ledger.add_prediction(
            handle, generation=generation, frame_idx=FRAME, time_sec=TIME,
            episode_revision=1, input_board=board, result=broken,
        )


def test_prediction_step_index_must_be_strict_one_based_sequence() -> None:
    ledger, handle, generation, board = _opened()
    result = _result(board)
    result.steps[0] = replace(result.steps[0], chain_index=13)
    with pytest.raises(LedgerValidationError):
        ledger.add_prediction(
            handle, generation=generation, frame_idx=FRAME, time_sec=TIME,
            episode_revision=1, input_board=board, result=result,
        )


def test_prediction_adjacent_step_boards_must_be_continuous() -> None:
    ledger = ChainPredictionLedger()
    generation, board = ChainGeneration("2P", 3, 11), _two_chain_board()
    handle = ledger.open_landing_provisional(
        generation=generation, frame_idx=FRAME, time_sec=TIME,
        origin_before_board=board, landing_event=_event(board),
    )
    result = _result(board)
    result.steps[1] = replace(result.steps[1], board_before=Board())
    with pytest.raises(LedgerValidationError):
        ledger.add_prediction(
            handle, generation=generation, frame_idx=FRAME, time_sec=TIME,
            episode_revision=1, input_board=board, result=result,
        )


@pytest.mark.parametrize(
    "field,change",
    [("total_erased", 1), ("total_ojama", 1), ("participating_cells", 1)],
)
def test_prediction_aggregate_must_match_steps(field: str, change: int) -> None:
    ledger, handle, generation, board = _opened()
    result = _result(board)
    broken = replace(result, **{field: getattr(result, field) + change})
    with pytest.raises(LedgerValidationError):
        ledger.add_prediction(
            handle, generation=generation, frame_idx=FRAME, time_sec=TIME,
            episode_revision=1, input_board=board, result=broken,
        )


def test_negative_score_delta_is_preserved_as_observation() -> None:
    ledger, handle, generation, _board_value = _opened()
    raw = ledger.record_raw_score(
        handle, generation=generation, frame_idx=FRAME, time_sec=TIME,
        source="score_reset_observation", raw_value=10,
        prev_score=999, cur_score=10, delta=-989, is_valid=True,
    )
    assert raw.delta == -989


@pytest.mark.parametrize(
    "kwargs",
    [
        {"raw_value": True}, {"prev_score": -1}, {"cur_score": True},
        {"delta": True}, {"is_valid": 1}, {"source": ""},
    ],
)
def test_invalid_raw_evidence_is_rejected_atomically(kwargs: dict[str, object]) -> None:
    ledger, handle, generation, board = _opened()
    values: dict[str, object] = {
        "source": "raw", "raw_value": 971, "prev_score": 970,
        "cur_score": 971, "delta": 1, "is_valid": True,
    }
    values.update(kwargs)
    with pytest.raises(LedgerValidationError):
        ledger.record_raw_score(
            handle, generation=generation, frame_idx=FRAME, time_sec=TIME, **values,
        )
    prediction = _add_origin_prediction(ledger, handle, generation, board)
    assert prediction.ledger_sequence == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"session_id": True}, {"session_id": -1}, {"step_index": True},
        {"step_index": -1}, {"total_power": -1}, {"step_product": True},
        {"source": "  "},
    ],
)
def test_invalid_formula_evidence_is_rejected_atomically(kwargs: dict[str, object]) -> None:
    ledger, handle, generation, board = _opened()
    values: dict[str, object] = {
        "source": "formula", "session_id": 2, "step_index": 0,
        "total_power": 0, "step_product": None,
    }
    values.update(kwargs)
    with pytest.raises(LedgerValidationError):
        ledger.record_formula(
            handle, generation=generation, frame_idx=FRAME, time_sec=TIME, **values,
        )
    prediction = _add_origin_prediction(ledger, handle, generation, board)
    assert prediction.ledger_sequence == 2


@pytest.mark.parametrize(
    "frame,time_value", [(True, TIME), (-1, TIME), (FRAME, True),
                          (FRAME, float("nan")), (FRAME, -0.1)],
)
def test_invalid_open_position_is_rejected(frame: object, time_value: object) -> None:
    ledger, board = ChainPredictionLedger(), _board()
    with pytest.raises(LedgerValidationError):
        ledger.open_landing_provisional(
            generation=ChainGeneration("2P", 3, 11), frame_idx=frame,
            time_sec=time_value, origin_before_board=board, landing_event=_event(board),
        )


def test_public_functions_respect_fifty_line_rule() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    too_long: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = int(node.end_lineno or node.lineno) - node.lineno + 1
            if length > 50:
                too_long.append((node.name, length))
    assert too_long == []
