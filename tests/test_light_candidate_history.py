"""同じ発火の候補を再参加させず、式の矛盾時は下限を撤回する。"""
from types import SimpleNamespace
import numpy as np
from src.light_likelihood import CandidateEvidence, LOG_RATIO
from src.light_refind import after_first, after_boards, PhysicalRefind
from src.fire_board import NativeTrace


def prepared() -> CandidateEvidence:
    board = np.zeros((13, 6), dtype=np.int8)
    candidates = [SimpleNamespace(board=board, steps=(40, 320), final=board),
                  SimpleNamespace(board=board, steps=(40, 320, 900), final=board)]
    result = CandidateEvidence(candidates)
    result.frames, result.first_t, result.last_t = 10, 0., .3
    return result


def test_excluded_short_candidate_does_not_return_when_mean_gap_shrinks() -> None:
    result = prepared()
    result.scores = np.array([-2 * LOG_RATIO, 0.]) * result.frames
    result.select((40,))
    assert result.retained == [1] and sum(result.steps) == 1260
    result.scores[:] = 0
    result.select((40,))
    assert result.retained == [1] and sum(result.steps) == 1260


def test_formula_contradiction_revokes_previously_published_bound() -> None:
    result = prepared()
    result.scores = np.array([-2 * LOG_RATIO, 0.]) * result.frames
    result.select((40,))
    result.select((40, 700))
    assert result.retained == [] and result.steps is None and result.final is None


def test_later_formula_does_not_revive_previously_excluded_candidate() -> None:
    result = prepared()
    result.scores = np.array([0., -2 * LOG_RATIO]) * result.frames
    result.select((40,))
    assert result.retained == [0]
    result.select((40, 320, 900))
    assert result.retained == [] and result.steps is None


def test_new_chain_has_independent_candidate_set() -> None:
    earlier, later = prepared(), prepared()
    earlier.scores = np.array([-2 * LOG_RATIO, 0.]) * earlier.frames
    earlier.select((40,))
    later.select((40,))
    assert earlier.retained == [1] and later.retained == [0, 1]


def test_observation_window_still_required_before_elimination() -> None:
    result = prepared()
    result.frames = 5
    result.scores = np.array([-2 * LOG_RATIO, 0.]) * result.frames
    result.select((40,))
    assert result.steps is None and result.viable.tolist() == [True, True]
    result.frames, result.scores = 10, np.zeros(2)
    result.select((40,))
    assert result.retained == [0, 1]


def hidden_stays_after_first() -> np.ndarray:
    board = np.zeros((13, 6), dtype=np.int8)
    board[1:, 0] = (1, 2, 3, 4) * 3
    board[12, 1:5] = 1
    return board


def test_unfallen_hidden_cell_remains_an_unknown_slot() -> None:
    board = hidden_stays_after_first()
    after, slots = after_first(board, 40, {0})
    assert slots == [(0, 0)]
    candidates = list(after_boards(board, set(), (0, 1, 2, 3, 4), 40))
    assert {int(c[0, 0]) for c in candidates} == {0, 1, 2, 3, 4}
    assert all(np.array_equal(c[1:], after[1:]) for c in candidates)


def test_visible_top_supports_candidates_without_inflow_flag() -> None:
    predictor = PhysicalRefind(NativeTrace())
    predictor.remember(.1, dict(state='STABLE', confirmed_board=hidden_stays_after_first()))
    predictor.start(1., 40, False)
    assert predictor.active and len(predictor.candidates) == 5


def test_empty_top_does_not_assume_hidden_cells() -> None:
    predictor = PhysicalRefind(NativeTrace())
    board = hidden_stays_after_first()
    board[1, 0] = 0
    predictor.remember(.1, dict(state='STABLE', confirmed_board=board))
    predictor.start(1., 40, False)
    assert not predictor.active and predictor.candidates == []
