"""元trackerの通常simulateで深さ制限の失敗を再現。原状態は変更しない。"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent


def load_join() -> Any:
    path = ROOT.parent / 'g2_fixed_side_join_2026-09-11_v1/fixed_join.py'
    spec = importlib.util.spec_from_file_location('_snapshot_original_join', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tracker() -> Any:
    from src.board import Board
    from src.chain_detector import VideoChainTracker
    value = VideoChainTracker()
    grid = [[0] * 6 for _ in range(13)]
    grid[12][:4] = [1] * 4
    board = Board.from_dict({'grid': grid})
    value._simulator.simulate(board)
    return value


def main() -> None:
    join, value = load_join(), tracker()
    try:
        join._vrepr(value)
    except join.SideJoinUninspectable as error:
        report = dict(error=str(error), tracker_type=type(value).__name__,
                      artificial_input=True, actual_J=False, G2=False)
        print(json.dumps(report))
        assert str(error) == 'max_depth:PuyoGroup'
    else:
        raise AssertionError('旧失敗が再現しない')


if __name__ == '__main__':
    main()
