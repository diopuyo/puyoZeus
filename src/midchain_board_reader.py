"""途中予測専用の画像読取。浮遊削除・確定履歴による穴埋めを行わない。"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from src.board import Board, COLOR_UNKNOWN, HIDDEN_ROWS
from src.board_state_machine import BoardState
from src.hybrid_classifier import DEFAULT_CNN_OVERRIDE_PROB

# E22レンダと同じ既存モデル。評価ごとのモデル切替はしない。
MODEL = Path('models/cnn_phase_b_large_v2.pt')


class MidchainBoardReader:
    """記録補完と動画レンダで同じ読取器を使う。"""

    def __init__(self) -> None:
        from src.recognition_pipeline import RecognitionPipeline
        self.reader = RecognitionPipeline._build_hybrid_reader(MODEL, cnn_override_prob=DEFAULT_CNN_OVERRIDE_PROB)
        self.reader._apply_inference = False

    def read(self, frame: np.ndarray, sides: tuple[Any, Any]) -> tuple[Board | None, Board | None]:
        """重力待ちの現フレームだけを読み、見えない上端はUNKNOWNとして拒否可能にする。"""
        boards: list[Board | None] = []
        regions = (self.reader._p1_region, self.reader._p2_region)
        for side, region in zip(sides, regions):
            if side.state != BoardState.GRAVITY_SETTLE:
                boards.append(None)
                continue
            board = self.reader.read_board(frame, region, skip_tier1=True)
            board._grid[:HIDDEN_ROWS] = np.where(board._grid[HIDDEN_ROWS] != 0, COLOR_UNKNOWN, 0)
            boards.append(board)
        return boards[0], boards[1]
