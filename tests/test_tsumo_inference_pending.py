"""確定を待たせず評価盤だけに適用する、仮の対の寿命を検証する。"""
from dataclasses import replace

import pytest

from src.board import Board
from src.tsumo_inference_pending import InferencePendingPair, MAX_PENDING_SEC
ORIGIN = ((1, 2, 1), (2, 2, 3))
MOVED = ((1, 3, 1), (2, 3, 3))


def board(cells: tuple = ()) -> Board:
    result = Board()
    result.set(12, 0, 9)
    result.set(11, 0, 2)
    for r, c, value in cells:
        result.set(r, c, value)
    return result


def project(tracker: InferencePendingPair, confirmed: Board, time: float,
            observed: Board | None = None, **kwargs: bool) -> Board:
    result = tracker.project(confirmed, observed or confirmed, time,
                             published=confirmed, next_shift=kwargs.get('next_shift', False),
                             active=kwargs.get('active', True))
    return result if result is not None else confirmed


def armed() -> tuple[InferencePendingPair, Board]:
    tracker = InferencePendingPair()
    project(tracker, board(), 0.0)
    confirmed = board(ORIGIN)
    evaluated = project(tracker, confirmed, 1.0)
    assert evaluated.get(1, 2) == 0
    return tracker, confirmed


def test_only_evaluation_board_changes() -> None:
    tracker, confirmed = armed()
    evaluated = project(tracker, confirmed, 1.1)
    assert confirmed.get(1, 2) == 1
    assert confirmed.get(2, 2) == 3
    assert evaluated.get(1, 2) == evaluated.get(2, 2) == 0
    assert evaluated.get(12, 0) == 9
    assert evaluated.get(11, 0) == 2


@pytest.mark.parametrize('elapsed', [0.0, 0.1, 0.5, 29/30])
def test_kept_out_before_deadline(elapsed: float) -> None:
    tracker, confirmed = armed()
    assert project(tracker, confirmed, 1.0+elapsed).get(1, 2) == 0


@pytest.mark.parametrize('elapsed', [MAX_PENDING_SEC, MAX_PENDING_SEC+0.1, 2.0])
def test_included_at_deadline_without_stopping_evaluation(elapsed: float) -> None:
    tracker, confirmed = armed()
    assert project(tracker, confirmed, 1.0+elapsed) is confirmed
    assert not tracker.pairs


def test_same_side_next_releases_immediately() -> None:
    tracker, confirmed = armed()
    assert project(tracker, confirmed, 1.1, next_shift=True) is confirmed


def test_next_on_addition_frame_cannot_confirm_the_new_pair() -> None:
    tracker = InferencePendingPair()
    project(tracker, board(), 0.0)
    confirmed = board(ORIGIN)
    assert project(tracker, confirmed, 1.0, next_shift=True).get(1, 2) == 0
    assert project(tracker, confirmed, 1.1, next_shift=True) is confirmed


@pytest.mark.parametrize('observed', [board(), board(MOVED)])
def test_disappeared_pair_is_not_resurrected_at_timeout(observed: Board) -> None:
    tracker, confirmed = armed()
    assert project(tracker, confirmed, 1.1, observed).get(1, 2) == 0
    assert project(tracker, confirmed, 3.0, observed).get(1, 2) == 0
    assert project(tracker, confirmed, 3.1, observed, next_shift=True).get(1, 2) == 0


def test_unknown_is_not_disappearance() -> None:
    tracker, confirmed = armed()
    unknown = board(((1, 2, 10), (2, 2, 10)))
    project(tracker, confirmed, 1.2, unknown)
    assert project(tracker, confirmed, 2.0) is confirmed


def test_other_side_does_not_release() -> None:
    left, left_board = armed()
    right, right_board = armed()
    project(right, right_board, 1.1, next_shift=True)
    assert project(left, left_board, 1.1).get(1, 2) == 0


def test_reset_at_match_boundary() -> None:
    tracker, confirmed = armed()
    assert project(tracker, confirmed, 1.1, active=False) is confirmed
    assert project(tracker, confirmed, 1.2) is confirmed


def test_time_rewind_clears_old_pairs() -> None:
    tracker, confirmed = armed()
    assert project(tracker, confirmed, 0.1) is confirmed


def test_reappearing_pair_keeps_original_deadline() -> None:
    tracker, confirmed = armed()
    assert project(tracker, confirmed, 1.2, board()).get(1, 2) == 0
    assert project(tracker, confirmed, 1.9).get(1, 2) == 0
    assert project(tracker, confirmed, 2.0) is confirmed


@pytest.mark.parametrize('state', ['ojama_fall', 'chain', 'gravity_settle', 'effect', 'menu'])
def test_other_phases_do_not_inherit_pair_masks(state: str) -> None:
    tracker, confirmed = armed()
    assert tracker.project(confirmed, board(), 1.2, published=confirmed,
                           active=True, next_shift=False, state=state) is None
    assert not tracker.pairs


@pytest.mark.parametrize('cells', [((11, 3, 1), (12, 3, 3)), ((1, 2, 1),),
    (*ORIGIN, (1, 4, 2)), ((1, 2, 9), (2, 2, 9))])
def test_not_an_upper_colored_pair(cells: tuple) -> None:
    tracker = InferencePendingPair()
    project(tracker, board(), 0.0)
    confirmed = board(cells)
    assert project(tracker, confirmed, 1.0) is confirmed


def test_changed_existing_cells_are_preserved() -> None:
    tracker, confirmed = armed()
    corrected = confirmed.copy()
    corrected.set(1, 2, 9)
    assert project(tracker, corrected, 1.1).get(1, 2) == 9


def test_discarded_fragment_stays_out_during_garbage() -> None:
    tracker, confirmed = armed()
    project(tracker, confirmed, 1.1, board())
    fragment = board((ORIGIN[0],))
    project(tracker, fragment, 1.2, board())
    result = tracker.project(fragment, board(), 1.3, published=fragment,
                             active=True, next_shift=False, state='ojama_fall')
    assert result is not None and result.get(1, 2) == 0
    assert fragment.get(1, 2) == 1


def test_replaced_pair_is_not_carried_into_garbage() -> None:
    tracker, confirmed = armed()
    corrected = confirmed.copy()
    corrected.set(2, 2, 5)
    project(tracker, corrected, 1.1, board())
    fragment = board((ORIGIN[0],))
    project(tracker, fragment, 1.2, board())
    assert tracker.project(fragment, board(), 1.3, published=fragment,
                           active=True, next_shift=False, state='ojama_fall') is None
    assert not tracker.pairs


def rotated_pair() -> tuple[InferencePendingPair, Board]:
    tracker, confirmed = armed()
    swapped = board(((1, 2, 3), (2, 2, 1)))
    project(tracker, confirmed, 1.1, swapped)
    project(tracker, confirmed, 1.2, swapped)
    return tracker, confirmed


@pytest.mark.parametrize('state', ['ojama_fall', 'chain', 'gravity_settle'])
def test_observed_rotation_keeps_old_cells_out_across_phases(state: str) -> None:
    tracker, confirmed = rotated_pair()
    result = tracker.project(confirmed, confirmed, 3.0, published=confirmed,
                             active=True, next_shift=True, state=state)
    assert result is not None
    assert result.get(1, 2) == result.get(2, 2) == 0
    assert confirmed.get(1, 2) == 1 and confirmed.get(2, 2) == 3


def test_one_swapped_observation_does_not_prove_rotation() -> None:
    tracker, confirmed = armed()
    project(tracker, confirmed, 1.1, board(((1, 2, 3), (2, 2, 1))))
    project(tracker, confirmed, 1.2, confirmed)
    assert tracker.project(confirmed, confirmed, 1.3, published=confirmed,
                           active=True, next_shift=False, state='chain') is None


@pytest.mark.parametrize('cells,swapped', [
    (((0, 2, 1), (1, 2, 3)), ((0, 2, 3), (1, 2, 1))),
    (((1, 2, 1), (1, 4, 3)), ((1, 2, 3), (1, 4, 1))),
])
def test_hidden_or_separate_cells_do_not_prove_rotation(cells: tuple, swapped: tuple) -> None:
    tracker = InferencePendingPair()
    project(tracker, board(), 0.0)
    confirmed = board(cells)
    project(tracker, confirmed, 1.0)
    project(tracker, confirmed, 1.1, board(swapped))
    project(tracker, confirmed, 1.2, board(swapped))
    assert tracker.project(confirmed, board(swapped), 1.3, published=confirmed,
                           active=True, next_shift=False, state='chain') is None


def test_rotated_cells_retire_individually_without_masking_new_pieces() -> None:
    tracker, confirmed = rotated_pair()
    changed = confirmed.copy()
    changed.set(2, 2, 5)
    result = tracker.project(changed, changed, 1.3, published=changed,
                             active=True, next_shift=False, state='chain')
    assert result.get(1, 2) == 0 and result.get(2, 2) == 5
    restored = confirmed.copy()
    result = tracker.project(restored, restored, 1.4, published=restored,
                             active=True, next_shift=False, state='chain')
    assert result.get(1, 2) == 0 and result.get(2, 2) == 3
    restored.set(1, 2, 5)
    assert tracker.project(restored, restored, 1.5, published=restored,
                           active=True, next_shift=False, state='chain') is None


def test_rotated_pair_clears_at_match_boundary() -> None:
    tracker, confirmed = rotated_pair()
    assert tracker.project(confirmed, confirmed, 1.3, published=confirmed,
                           active=False, next_shift=False, state='menu') is None
    assert not tracker.pairs


def test_public_color_correction_does_not_bypass_pending_pair() -> None:
    tracker, confirmed = armed()
    published = confirmed.copy()
    published.set(1, 2, 4)
    evaluated = tracker.project(confirmed, confirmed, 1.1, published=published,
                                active=True, next_shift=False)
    assert evaluated.get(1, 2) == evaluated.get(2, 2) == 0
    assert published.get(1, 2) == 4
    assert confirmed.get(1, 2) == 1


def test_new_garbage_at_publication_is_preserved() -> None:
    tracker, confirmed = armed()
    published = confirmed.copy()
    published.set(1, 2, 9)
    evaluated = tracker.project(confirmed, confirmed, 1.1, published=published,
                                active=True, next_shift=False)
    assert evaluated.get(1, 2) == 9


def test_projection_uses_existing_evaluation_delivery() -> None:
    from src.recognition_pipeline import SideResult, PipelineResult
    from src.board_state_machine import BoardState
    from src.row1_publication import evaluation_result, publication_result
    tracker, confirmed = armed()
    # 実dataclassの必要引数を使い、評価後も公開盤を復元できることを確かめる。
    side = SideResult(side='1P', state=BoardState.STABLE, cnn_board=confirmed,
        inferred_board=None, confirmed_board=confirmed, drift=None,
        score=0, score_delta=0, chain_event=None,
        evaluation_confirmed_board=project(tracker, confirmed, 1.1))
    result = PipelineResult(frame_idx=1, time_sec=1.1, is_match_active=True,
                            p1=side, p2=replace(side, side='2P'))
    evaluated = evaluation_result(result)
    assert evaluated.p1.confirmed_board.get(1, 2) == 0
    assert result.p1.confirmed_board.get(1, 2) == 1
    assert publication_result(evaluated, result).p1.confirmed_board is confirmed


def test_flag_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_speed import recognition_config
    monkeypatch.delenv('PUYO_W68_INFERENCE_PENDING', raising=False)
    assert 'enable_w68_inference_pending' not in recognition_config({}, True)
    monkeypatch.setenv('PUYO_W68_INFERENCE_PENDING', '1')
    assert recognition_config({}, True)['enable_w68_inference_pending'] is True


def test_publication_guard_keeps_pending_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.recognition_pipeline import RecognitionPipeline, SideResult, PipelineResult
    from src.board_state_machine import BoardState
    tracker, confirmed = armed()
    filtered = project(tracker, confirmed, 1.1)
    side = SideResult(side='1P', state=BoardState.STABLE, cnn_board=confirmed,
        inferred_board=None, confirmed_board=confirmed, drift=None,
        score=0, score_delta=0, chain_event=None, evaluation_confirmed_board=filtered)
    result = PipelineResult(frame_idx=1, time_sec=1.1, is_match_active=True,
                            p1=side, p2=replace(side, side='2P'))
    pipeline = RecognitionPipeline.__new__(RecognitionPipeline)
    monkeypatch.setattr(pipeline, '_apply_row1_col2_guard', lambda *args, **kwargs: confirmed.copy())
    monkeypatch.setattr(pipeline, '_row1_formula_visible', lambda side: False)
    published = pipeline._publish_row1_result(result)
    assert published.p1.evaluation_confirmed_board is filtered
    assert published.p1.confirmed_board.get(1, 2) == 1


def test_release_default_on() -> None:
    """2026-11-12 user 採用で 0.6.5 の配布は既定 ON。明示の '0' で OFF にできる。"""
    from src.phase_j.release_config import release_environment
    assert release_environment({})['PUYO_W68_INFERENCE_PENDING'] == '1'
    assert release_environment({'PUYO_W68_INFERENCE_PENDING': '0'})['PUYO_W68_INFERENCE_PENDING'] == '0'
