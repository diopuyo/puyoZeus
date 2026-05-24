"""孤立 cell の周囲色補正 (Phase Z 試行 A)。

「ぷよぷよは puyo が連結してフィールドを構成する」性質を利用:
- 自身が puyo 色、4 近傍の puyo cell が全て別色 → 孤立色 1 cell = 誤検出候補
- 周囲 ≥ N cell が同色 (自身と異色) なら、その色に補正

連鎖中・落下中は正常な孤立 cell が出るので、is_chain=True 時 skip。

設計:
    - 各 puyo cell について 4 近傍 (上下左右) を確認
    - 上下左右の puyo cell (EM/UNKNOWN を除く) を集計
    - 周囲 puyo ≥ MIN_NEIGHBOR が同色で、自身がそれと異なる puyo 色 → 補正
    - EM cell は変更しない (検出漏れではなく真の空)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN,
    HIDDEN_ROWS, Board,
)

# 周囲 puyo ≥ N cell が同色なら補正
MIN_NEIGHBOR_SAME_COLOR: int = 3
# 補正対象は puyo 色 (R/B/G/Y/P) のみ、OJM は対象外 (孤立 OJM はある)
TRAINABLE_COLORS: tuple[int, ...] = (1, 2, 3, 4, 5)  # RED/BLUE/GRN/YEL/PUR


@dataclass
class ConnectivityOutlierRefiner:
    """孤立 cell の周囲色補正。"""

    min_neighbor_same: int = MIN_NEIGHBOR_SAME_COLOR

    def refine(
        self,
        board: Board,
        is_chain: bool = False,
    ) -> tuple[Board, np.ndarray]:
        """周囲と異色の puyo cell を補正。

        Returns:
            (refined_board, outlier_mask shape=(BOARD_ROWS, BOARD_COLS) bool)
        """
        out = board.copy()
        outlier_mask = np.zeros(
            (BOARD_ROWS, BOARD_COLS), dtype=bool,
        )
        if is_chain:
            return out, outlier_mask
        # 隠し段は周囲情報少ないため可視段のみ対象
        for vrow in range(12):
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                color = int(board.get(row, col))
                if color not in TRAINABLE_COLORS:
                    continue
                # 4 近傍の puyo 色を集計
                neighbor_counts: dict[int, int] = {}
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = row + dr
                    nc = col + dc
                    if not (0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS):
                        continue
                    nc_color = int(board.get(nr, nc))
                    if nc_color in (COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA):
                        continue
                    neighbor_counts[nc_color] = (
                        neighbor_counts.get(nc_color, 0) + 1
                    )
                if not neighbor_counts:
                    continue
                # 最頻色 (自身と違う色)
                max_color, max_count = max(
                    neighbor_counts.items(), key=lambda kv: kv[1],
                )
                if (max_color != color
                        and max_count >= self.min_neighbor_same):
                    out.set(row, col, max_color)
                    outlier_mask[row, col] = True
        return out, outlier_mask


__all__ = [
    "ConnectivityOutlierRefiner",
    "MIN_NEIGHBOR_SAME_COLOR",
    "TRAINABLE_COLORS",
]
