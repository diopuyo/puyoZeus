"""
時間方向スムージング (連続フレームの per-cell 最頻色で盤面を安定化)。

CNN 単発予測は連鎖 halo / UI overlay / 瞬間的エフェクトで誤分類することがある。
それらは通常 1-2 フレームだけ発生するため、N フレームの多数決で消える。
本モジュールは per-cell ringbuffer + モード集計で 0.5 秒ウィンドウの
「安定盤面」を返す。

想定ユースケース:
    smoother = TemporalSmoother(window_size=15)  # 0.5s @ 30fps
    for board in boards_per_frame:
        stable_board = smoother.update(board)
    # stable_board は「直近 window_size フレームの per-cell 最頻色」

評価用途:
    15 フレームのうち 12 フレームで CNN が赤と読み、halo で 3 フレームだけ
    緑と読んだ場合 → 安定化後は赤。halo 誤認を消せる。
"""
from __future__ import annotations

from collections import deque

import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    Board,
)


DEFAULT_WINDOW_SIZE: int = 15  # 0.5 秒 × 30fps


class TemporalSmoother:
    """
    N フレームの盤面履歴を保持し、per-cell 最頻色で安定化する。

    Attributes:
        window_size: 履歴に保持するフレーム数。
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        """
        Args:
            window_size: 履歴保持フレーム数。0.5 秒窓なら fps × 0.5。
                最低 1。1 ならスムージング無効 (そのまま返す)。
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self._window_size = window_size
        self._history: deque[np.ndarray] = deque(maxlen=window_size)

    @property
    def window_size(self) -> int:
        return self._window_size

    def __len__(self) -> int:
        """現在の履歴フレーム数。"""
        return len(self._history)

    def reset(self) -> None:
        """履歴を空にする。別動画/別セグメントの開始時に呼ぶ。"""
        self._history.clear()

    def update(self, board: Board) -> Board:
        """
        新しい盤面を履歴に積んで、安定化後の盤面を返す。

        履歴が window_size に満たないうちは手元の履歴で多数決する
        (コールドスタート対応)。

        Args:
            board: この時点で CNN が予測した盤面。

        Returns:
            Board: 履歴 per-cell 最頻色で再構成された盤面。
        """
        self._history.append(board._grid.copy())
        stable_grid = self._compute_majority()
        result = Board()
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                result.set(row, col, int(stable_grid[row, col]))
        return result

    def _compute_majority(self) -> np.ndarray:
        """履歴から per-cell の最頻色を計算する。"""
        if not self._history:
            # ありえないが念のため: 空履歴では全 0 (empty) を返す
            return np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.uint8)
        stack = np.stack(list(self._history), axis=0)  # shape (N, 13, 6)
        # per-cell に bincount で最頻色を取る。
        # クラス値は 0〜9 の範囲 (COLOR_OJAMA=9) なので minlength=10 で十分。
        return self._mode_per_cell(stack)

    @staticmethod
    def _mode_per_cell(stack: np.ndarray) -> np.ndarray:
        """
        shape (N, R, C) → shape (R, C) per-cell 最頻値。

        同票時は小さい値 (argmax が最初に見つけた値) を返す。
        """
        _, rows, cols = stack.shape
        result = np.zeros((rows, cols), dtype=np.uint8)
        # bincount はスカラー入力のみなので per-cell ループする。
        # 78 セル × N フレームは無視できるコスト。
        max_value = 10  # 0-9 を想定 (余裕を持って 10)
        for r in range(rows):
            for c in range(cols):
                col_values = stack[:, r, c]
                counts = np.bincount(col_values, minlength=max_value)
                result[r, c] = int(np.argmax(counts))
        return result
