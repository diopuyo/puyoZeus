"""D4: 操作中の色一致を固定セルへ昇格させない。"""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from src.board import Board
from src.board_state_machine import StateContext, DetectorSignals, _apply_stable_recovery_gate
from src.placement_inferrer import apply_persistent_landing_color_guard


@pytest.mark.parametrize('enabled', [False, True])
def test_recovery_requires_placement(enabled: bool) -> None:
    board = Board()
    board._grid[2:, 2] = 9
    observed = board.copy()
    observed.set(1, 2, 5)
    ctx = StateContext(confirmed_board=board)
    signals = DetectorSignals(is_match_active=True, time_sec=1., cnn_board=observed, hsv_board=observed)
    _apply_stable_recovery_gate(ctx, signals, 1, falling_puyo_integrity=enabled)
    assert ctx.confirmed_board.get(1, 2) == (0 if enabled else 5)


def test_completed_placement_can_recover() -> None:
    board, observed = Board(), Board()
    observed.set(12, 2, 3)
    ctx = StateContext(confirmed_board=board)
    signals = DetectorSignals(is_match_active=True, time_sec=1., cnn_board=observed, hsv_board=observed,
                              slide_motion=True, placement_validated=True)
    _apply_stable_recovery_gate(ctx, signals, 1, falling_puyo_integrity=True)
    assert ctx.confirmed_board.get(12, 2) == 3


def test_ojama_recovery_is_not_operated_pair() -> None:
    board, observed = Board(), Board()
    observed.set(12, 2, 9)
    ctx = StateContext(confirmed_board=board)
    signals = DetectorSignals(is_match_active=True, time_sec=1., cnn_board=observed, hsv_board=observed)
    _apply_stable_recovery_gate(ctx, signals, 1, falling_puyo_integrity=True)
    assert ctx.confirmed_board.get(12, 2) == 9


@pytest.mark.parametrize('enabled', [False, True])
def test_watch_respects_grace_removal(enabled: bool) -> None:
    board, observed = Board(), Board()
    observed.set(1, 2, 3)
    region = NS(cell_sample_rect=lambda r, c: (0, 0, 4, 4))
    result, _ = apply_persistent_landing_color_guard(board, [(1, 2)], observed,
        NS(classify=lambda p: 3), np.zeros((4, 4, 3), dtype=np.uint8), region,
        falling_puyo_integrity=enabled)
    assert result.get(1, 2) == (0 if enabled else 3)
