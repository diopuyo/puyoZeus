"""人工正常手で着地→原origin→連鎖→次手を同じjoint状態へ結ぶCPU対照。"""
from __future__ import annotations
from pathlib import Path
import sys
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_belief_hidden_landing_2026-09-11_v1'))
import transition_candidates as T

B, H = T.B, T.H
SCOPE = ('fixture-source', 'fixture-run', 3, 11, 12, 3, '1P')
PRIOR = H.uncalibrated_uniform('列挙配置は未較正一様。実際の着手頻度ではない。')


def source() -> tuple[Any, Any, Any]:
    board = B.Board()
    for row in range(10, 13): board.set(row, 0, 1)
    for row in range(1, 13): board.set(row, 5, 2 if row % 2 else 3)
    board.set(0, 5, 10)
    pb = B.ProbabilisticBoard.from_board(board)
    pb.set_distribution(0, 5, {4: 0.3, 5: 0.7})
    value = B.establish(SCOPE, 10, 30, board, pb)
    origin = board.copy()
    origin.set(8, 0, 4)
    origin.set(9, 0, 1)
    point = B.Board.from_dict({'grid': value.worlds[0].grid})
    point.set(8, 0, 4)
    point.set(9, 0, 1)
    result = B.ChainSimulator(exclude_hidden_row_from_pop=True).simulate(point)
    assert result.chain_count == 1
    return value, origin, result.final_board


def test_one_call_landing_chain_and_next_hand_keeps_joint() -> None:
    value, origin, final = source()
    following, report = T.run(value, SCOPE, 12, 'actual-token-assumed-fixture-1',
        (1, 4), final, 1, PRIOR, origin_observed=origin)
    assert following.frame == 12 and len(following.worlds) == 2
    assert B.marginals(following).cell(0, 5).probs == pytest.approx({4: 0.3, 5: 0.7})
    assert report.chain_count == 1 and not report.physical_certified
    next_board = final.copy()
    next_board.set(10, 0, 2)
    next_board.set(11, 0, 3)
    last, _ = T.run(following, SCOPE, 14, 'actual-token-assumed-fixture-2', (2, 3), next_board, 0, PRIOR)
    assert len(last.tokens) == 2 and last.frame == 14
    assert B.marginals(last).cell(0, 5).probs == pytest.approx({4: 0.3, 5: 0.7})
    assert value.tokens == () and not last.quality_gate_clear


def test_separate_same_frame_operations_remain_rejected() -> None:
    value, origin, final = source()
    landed, _ = B.land(value, SCOPE, 12, 'land', (1, 4), ((8, 0, 4), (9, 0, 1)), origin)
    with pytest.raises(ValueError, match='deadline_or_clock'):
        B.settle(landed, SCOPE, 12, 'chain', 1, final)
    combined, _ = T.run(value, SCOPE, 12, 'single-consumption', (1, 4), final, 1, PRIOR, origin_observed=origin)
    assert combined.tokens == ('single-consumption',)


@pytest.mark.parametrize('case', ('missing_origin', 'wrong_origin', 'wrong_chain', 'wrong_final', 'same_frame'))
def test_invalid_transition_preserves_input(case: str) -> None:
    value, origin, final = source()
    frame, chain = 12, 1
    if case == 'missing_origin': origin = None
    elif case == 'wrong_origin': origin.set(12, 1, 5)
    elif case == 'wrong_chain': chain = 2
    elif case == 'wrong_final': final.set(12, 1, 5)
    else: frame = 10
    with pytest.raises(ValueError):
        T.run(value, SCOPE, frame, 'one', (1, 4), final, chain, PRIOR, origin_observed=origin)
    assert value.frame == 10 and value.tokens == () and len(value.worlds) == 2
