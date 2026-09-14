"""3過去callだけの候補を原SM/mergeと人工境界で検査する。"""
from __future__ import annotations
import contextlib
from types import SimpleNamespace as N
from typing import Any
import pytest
from test_endpoint_votes import objects, DetectorSignals, BoardState
import prior_votes_v2 as V
from src.board_state_machine import _vote_majority_board, _merge_diff_only


def fixture() -> tuple[Any, list[Any], list[Any]]:
    sm, raw = objects()
    history = [raw.copy()]
    phases = [('stable', 'stable'), ('stable', 'tsumo_fall'), ('tsumo_fall', 'tsumo_fall'), ('tsumo_fall', None)]
    ticks = [V.Tick(i * 2, *phase, raw.copy(), True, i * 2 / 60, (4, 5))
             for i, phase in enumerate(phases)]
    ticks[2].history_tail_id = id(history[0])
    return sm, ticks, history


def test_original_sm_recovers_without_exit_vote() -> None:
    sm, raw = objects()
    with contextlib.ExitStack() as stack:
        value = V.install(stack, sm, lambda: True)
        for frame in range(34946, 34954, 2):
            result = sm.update(frame, DetectorSignals(frame / 60, raw.copy(), True,
                next_pair=(4, 5), hsv_board=raw.copy(), effect_gate_window_active=False))
        assert result.state == BoardState.STABLE
        assert [result.confirmed_board.get(r, 4) for r in (1, 2)] == [4, 3]
        assert value.rows[0]['frames'] == [34946, 34948, 34950]
        assert value.rows[0]['frame'] == 34952 and not value.rows[0]['exit_used_as_vote']


@pytest.mark.parametrize('majority', [False, True])
def test_different_exit_does_not_select_past_votes(majority: bool) -> None:
    sm, ticks, history = fixture()
    ticks[-1].board.set(2, 4, 5)
    prior = V.past_history(ticks, history)
    assert prior is not None and len(prior) == 3
    assert all(board.get(2, 4) == 3 for board in prior)
    guard = _vote_majority_board(prior, min_votes=3)
    result = _merge_diff_only(sm.context.confirmed_board, ticks[-1].board,
                              empty_to_color_guard=guard, merge_use_majority_value=majority)
    assert result.get(2, 4) == (3 if majority else 0)
    assert result != ticks[-1].board  # 原mergeの不一致を後段の可視一致gateで救済しない。


def test_original_sm_different_exit_is_not_a_vote() -> None:
    sm, raw = objects()
    with contextlib.ExitStack() as stack:
        value = V.install(stack, sm, lambda: True)
        for frame in range(34946, 34954, 2):
            current = raw.copy()
            if frame == 34952:
                current.set(2, 4, 5)
            result = sm.update(frame, DetectorSignals(frame / 60, current, True,
                next_pair=(4, 5), hsv_board=current.copy(), effect_gate_window_active=False))
        # 生入力が変わると原detectorが退出自体を止める。票で退出を強制しない。
        assert result.state == BoardState.TSUMO_FALL
        assert value.rows == []
        assert result.confirmed_board.get(2, 4) == 0


@pytest.mark.parametrize('fault', ['normal_moving', 'duplicate', 'clock', 'next', 'history_identity', 'effect'])
def test_invalid_or_uncovered_window_is_unchanged(fault: str) -> None:
    _, ticks, history = fixture()
    if fault == 'normal_moving': ticks[0].board.set(2, 4, 0)
    if fault == 'duplicate': ticks[1].frame = 0
    if fault == 'clock': ticks[1].clock = 3
    if fault == 'next': ticks[1].next_pair = (2, 2)
    if fault == 'history_identity': history = [history[0].copy()]
    if fault == 'effect': ticks[1].quiet = False
    assert V.past_history(ticks, history) is None


def test_original_failure_poisoned_and_preserved() -> None:
    sm, raw = objects()
    error = ValueError('原detectorの失敗')
    def fail(*args: Any) -> None:
        raise error
    sm._detectors = [N(detect=fail)]
    with contextlib.ExitStack() as stack:
        value = V.install(stack, sm, lambda: True)
        with pytest.raises(ValueError) as caught:
            sm.update(0, DetectorSignals(0, raw, True))
        assert caught.value is error and value.failure is error and value.rows == []
        with pytest.raises(RuntimeError, match='prior_failure'):
            sm.update(2, DetectorSignals(2 / 60, raw, True))
