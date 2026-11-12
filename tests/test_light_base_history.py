"""可視盤の実観測から追加候補を作る条件を確かめる。"""
import numpy as np
import pytest
from src.light_base_history import ObservedBases


def recorded(board: np.ndarray) -> ObservedBases:
    history = ObservedBases()
    for t in np.linspace(0,.3,6):
        history.remember(float(t),board)
    return history


def test_repeated_visible_board_is_added_without_replacing_base() -> None:
    base = np.zeros((13,6),dtype=np.int8)
    raw = base.copy()
    raw[12,0] = 2
    variants = recorded(raw).variants(.3,base)
    assert len(variants) == 2
    assert np.array_equal(variants[0],base)
    assert np.array_equal(variants[1],raw)
    assert not base.any()


@pytest.mark.parametrize('end,count',[(.3,5),(.2,6)])
def test_short_observation_does_not_add_a_board(end: float, count: int) -> None:
    history = ObservedBases()
    base = np.zeros((13,6),dtype=np.int8)
    raw = base.copy()
    raw[12,0] = 2
    for t in np.linspace(0,end,count):
        history.remember(float(t),raw)
    assert len(history.variants(end,base)) == 1


def test_old_or_future_observation_is_not_used() -> None:
    raw = np.zeros((13,6),dtype=np.int8)
    raw[12,0] = 2
    history = recorded(raw)
    assert len(history.variants(5,np.zeros_like(raw))) == 1
    assert len(history.variants(-1,np.zeros_like(raw))) == 1


def test_known_color_conflict_is_not_used() -> None:
    raw = np.zeros((13,6),dtype=np.int8)
    raw[12,0] = 2
    base = raw.copy()
    base[12,0] = 1
    assert len(recorded(raw).variants(.3,base)) == 1


def test_falling_piece_is_not_a_stable_base() -> None:
    raw = np.zeros((13,6),dtype=np.int8)
    raw[10,0] = 2
    assert len(recorded(raw).variants(.3,np.zeros_like(raw))) == 1


def test_hidden_row_is_never_added() -> None:
    raw = np.zeros((13,6),dtype=np.int8)
    raw[0,0] = 2
    assert len(recorded(raw).variants(.3,np.zeros_like(raw))) == 1


def test_all_repeated_boards_are_retained() -> None:
    base = np.zeros((13,6),dtype=np.int8)
    history = ObservedBases()
    for color in (1,2):
        raw = base.copy()
        raw[12,0] = color
        for offset in np.linspace(0,.3,6):
            history.remember(color+float(offset),raw)
    variants = history.variants(2.3,base)
    assert [board[12,0] for board in variants] == [0,1,2]
