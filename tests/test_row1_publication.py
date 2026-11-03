"""公開補正と評価入力の分離、および先行NEXTと着地の対応を検証する。"""
from dataclasses import replace
import pickle

import pytest

from src.board import Board
from src.board_state_machine import BoardState
from src.recognition_pipeline import RecognitionPipeline, PipelineResult, SideResult
from src.row1_guard import Row1Guard, NEXT_LANDING_WINDOW_SEC
from src.row1_publication import evaluation_result, publication_result
from src.phase_j.live_bridge import RecognitionNotice, PipelineView
from src.phase_j.live_process import NoticeDeltaCodec


def side(top: int = 0) -> SideResult:
    board = Board()
    board.set(2, 2, 3)
    board.set(1, 2, top)
    board.set(0, 2, top)
    return SideResult('1P', BoardState.STABLE, board, board, board, None, 0, 0, None,
                      raw_cnn_board=board)


def test_publication_and_delta_preserve_evaluation() -> None:
    pipe = RecognitionPipeline.__new__(RecognitionPipeline)
    pipe._enable_row1_col2_guard, pipe._row1_guards = True, {}
    empty = PipelineResult(0, 0., True, side(), replace(side(), side='2P'))
    pipe._publish_row1_result(empty)
    internal = replace(empty, frame_idx=1, time_sec=.1, p1=side(3))
    published = pipe._publish_row1_result(internal)
    assert internal.p1.confirmed_board.get(1, 2) == 3
    assert published.p1.confirmed_board.get(1, 2) == 0
    assert pickle.dumps(evaluation_result(published)) == pickle.dumps(internal)
    view = PipelineView((1, 2), (None, None), (0, 0), (False, False))
    notice = RecognitionNotice(1, .1, 0., 0., pickle.dumps(published), view, 0)
    decoded = NoticeDeltaCodec().decode(NoticeDeltaCodec().encode(notice))
    assert decoded.result().p1.confirmed_board.get(1, 2) == 0
    assert decoded.evaluation_result().p1.confirmed_board.get(1, 2) == 3
    assert publication_result(internal, published).p1.confirmed_board.get(1, 2) == 0


@pytest.mark.parametrize('elapsed,settled,raw_matches,expected', [
    (7/30, True, True, 3), (7/30, False, True, 0),
    (NEXT_LANDING_WINDOW_SEC+.1, True, True, 0), (7/30, True, False, 0),
])
def test_preceding_next_requires_fresh_landing_and_matching_raw(
    elapsed: float, settled: bool, raw_matches: bool, expected: int,
) -> None:
    guard = Row1Guard()
    empty, full = side().confirmed_board, side(3).confirmed_board
    guard.step(empty, empty, 0., next_moved=True)
    result = guard.step(full, full if raw_matches else empty, elapsed, settled=settled)
    assert result.get(1, 2) == expected


def test_next_is_consumed_by_first_landing_even_without_target() -> None:
    guard = Row1Guard()
    empty, full = side().confirmed_board, side(3).confirmed_board
    guard.step(empty, empty, 0., next_moved=True)
    guard.step(empty, empty, .1, settled=True)
    assert guard.step(full, full, .2, settled=True).get(1, 2) == 0


def test_off_projection_returns_same_objects() -> None:
    result = PipelineResult(0, 0., True, side(), side())
    assert evaluation_result(result) is result
    assert publication_result(result, result) is result
