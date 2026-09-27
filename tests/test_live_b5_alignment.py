"""整列は時刻差と色差を混同せず、未対応を残す。"""
import numpy as np
from scripts.analyze_live_b5_alignment import pair_events, align


def event(frame: int, cells: dict) -> dict:
    board = np.zeros((13, 6), dtype=int)
    for (row, col), color in cells.items():
        board[row, col] = color
    return dict(board=board, frames=[frame], epoch=0)


def test_hidden_difference_does_not_prevent_turn_matching() -> None:
    a = event(1, {(12, 0): 1})
    b = event(2, {(12, 0): 1, (0, 0): 3})
    assert pair_events([a], [b]) == [(0, 0, 'visible_exact')]


def test_single_color_error_between_anchors_survives_alignment() -> None:
    empty, end = event(0, {}), event(20, {(12, 1): 2})
    a, b = event(10, {(12, 0): 2}), event(11, {(12, 0): 9})
    pairs = pair_events([empty, a, end], [empty, b, end])
    assert [(i, j) for i, j, _ in pairs] == [(0, 0), (1, 1), (2, 2)]
    assert 'one_color' in pairs[1][2]


def test_empty_puyo_difference_is_not_assumed_same_turn() -> None:
    assert pair_events([event(1, {(12, 0): 1})], [event(1, {})]) == []


def test_far_or_other_match_cannot_be_paired() -> None:
    a, b = event(0, {}), event(100, {})
    assert pair_events([a], [b]) == []
    b['frames'], b['epoch'] = [0], 1
    assert pair_events([a], [b]) == []


def test_partition_preserves_original_denominator() -> None:
    a = dict(boards=np.zeros((4, 2, 13, 6), dtype=int), stable=np.ones((4, 2), dtype=bool),
             active=np.ones(4, dtype=bool))
    b = {key: value.copy() for key, value in a.items()}
    a['boards'][1:, 0, 12, 0] = 2
    b['boards'][2:, 0, 12, 0] = 2
    result = align(a, b)
    assert result['original_different_cells'] == 1
    assert result['timing_cells'] == 1 and result['persistent_color_cells'] == 0
