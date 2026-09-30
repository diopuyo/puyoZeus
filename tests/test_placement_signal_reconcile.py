"""R1の合図選択・安全条件・故障注入。"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pytest
from src.board import Board
from src.placement_signal_reconcile import Observation, reconcile, select_observation


def observation(frame: int = 10) -> Observation:
    """底に赤を一つ置いた独立一致画像。"""
    grid = Board()._grid
    grid[12, 0] = 1
    return Observation(frame, frame/60, grid.copy(), grid.copy())


@pytest.mark.parametrize('signal,expected', [('next', 10), ('ojama', 9), ('formula', 9)])
def test_signal_frame(signal: str, expected: int) -> None:
    history = [observation(i) for i in range(5, 11)]
    assert select_observation(history, signal, 10).frame == expected


def test_formula_max_agreement_and_no_future() -> None:
    history = [observation(i) for i in range(5, 12)]
    for obs in history:
        if obs.frame != 7:
            obs.hsv[12, 0] = 2
    assert select_observation(history, 'formula', 10).frame == 7


def test_correct_without_mutating_input_or_hidden_row() -> None:
    board, obs = Board(), observation()
    board.set(0, 0, 5)
    fixed, audit = reconcile(board, obs, observation(9))
    assert fixed.get(12, 0) == 1 and fixed.get(0, 0) == 5
    assert board.get(12, 0) == 0
    assert len(audit['corrections']) == 1


@pytest.mark.parametrize('quality', ['smoke', 'burst', 'all_clear', 'flash'])
@pytest.mark.parametrize('previous_bad', [False, True])
def test_quality_never_corrects(quality: str, previous_bad: bool) -> None:
    obs, previous = observation(), observation(9)
    if previous_bad:
        previous = replace(previous, quality=quality)
    else:
        obs = replace(obs, quality=quality)
    assert reconcile(Board(), obs, previous)[0] is None


@pytest.mark.parametrize('erasing_current', [False, True])
def test_erasure_block(erasing_current: bool) -> None:
    obs, previous = observation(), observation(9)
    if erasing_current:
        obs = replace(obs, erasing=True)
    else:
        previous = replace(previous, erasing=True)
    assert reconcile(Board(), obs, previous)[0] is None


@pytest.mark.parametrize('frame', [None, 8, 10, 11])
def test_nonconsecutive(frame: int | None) -> None:
    previous = observation(frame) if frame is not None else None
    assert reconcile(Board(), observation(), previous)[1]['reason'] == 'nonconsecutive'


@pytest.mark.parametrize('count,accepted', [(6, True), (7, False)])
def test_difference_limit(count: int, accepted: bool) -> None:
    obs = observation()
    obs.cnn[12] = obs.hsv[12] = 1
    if count == 7:
        obs.cnn[11, 0] = obs.hsv[11, 0] = 1
    previous = replace(obs, frame=9)
    fixed, audit = reconcile(Board(), obs, previous)
    assert (fixed is not None) == accepted
    assert len(audit['differences']) == count


@pytest.mark.parametrize('kind', ['hsv_disagrees', 'previous_disagrees', 'unknown'])
def test_cell_evidence(kind: str) -> None:
    obs, previous = observation(), observation(9)
    if kind == 'hsv_disagrees':
        obs.hsv[12, 0] = 2
    elif kind == 'previous_disagrees':
        previous.cnn[12, 0] = previous.hsv[12, 0] = 2
    else:
        obs.cnn[12, 0] = obs.hsv[12, 0] = 10
    assert reconcile(Board(), obs, previous)[0] is None


def test_floating_rejected_without_deleting_other_cells() -> None:
    obs = observation()
    obs.cnn[10, 0] = obs.hsv[10, 0] = 3
    assert reconcile(Board(), obs, replace(obs, frame=9))[1]['reason'] == 'floating'


def test_removal_and_support_added_atomically() -> None:
    board, obs = Board(), observation()
    board.set(12, 1, 3)
    obs.cnn[11, 0] = obs.hsv[11, 0] = 2
    fixed, audit = reconcile(board, obs, replace(obs, frame=9))
    assert fixed.get(12, 1) == 0 and fixed.get(11, 0) == 2
    assert len(audit['corrections']) == 3


@pytest.mark.parametrize('missing', ['board', 'frame'])
def test_missing_inputs(missing: str) -> None:
    fixed, audit = reconcile(None if missing == 'board' else Board(),
                             None if missing == 'frame' else observation(), observation(9))
    assert fixed is None and audit['reason'] == f'missing_{missing}'


def test_no_frame_in_formula_window() -> None:
    assert select_observation([observation(5), observation(10)], 'formula', 10) is None
