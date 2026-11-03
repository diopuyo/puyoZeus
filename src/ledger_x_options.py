"""台帳と火力の分離を検証する未採用の旗。製品の既定値は変えない。"""
from __future__ import annotations

import os

PARTIAL_ENV = 'PUYO_LEDGER_PARTIAL'
BOARD_ENV = 'PUYO_X_BOARD_ONLY'
BOUNDARY_ENV = 'PUYO_X_B3_BOUNDARY'


def partial_enabled() -> bool:
    return os.environ.get(PARTIAL_ENV, '0') == '1'


def board_enabled() -> bool:
    return os.environ.get(BOARD_ENV, '0') == '1'


def boundary_game(frame: dict) -> int:
    """採用番号が未到達の間だけ既存の局所番号を維持する。"""
    if os.environ.get(BOUNDARY_ENV, '0') == '1' and frame.get('adopted_game') is not None:
        return int(frame['adopted_game'])
    return int(frame['game_idx'])
