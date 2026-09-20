"""W7-N3: 孤立した同色 1 セルを UNKNOWN 化。

V2.3 ConnectivityShapeRefiner は「異色 1 セルを多数色に補正」する正方向。
本フィルタはその逆: 「自色と異なる通常色 (RED..PURPLE) に完全に囲まれた 1 セル」
を誤認の可能性大として UNKNOWN にする。

例:
    周囲 RBR
        Y .
        RBR
    中央 Y は周囲 RB 多数色なので誤認の可能性 → UNKNOWN

設計上の注意:
    - 自色は通常色 (EMPTY/UNKNOWN/OJAMA は対象外)
    - 隣接 4 セルすべてが「自色と異なる通常色」のときのみ補正 (盤面端は隣接数少なく安全側)
    - V2.3 と相互作用しないよう「自色とは違う色が囲んでる」のみが条件
"""
from __future__ import annotations

from dataclasses import dataclass

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    Board,
)

# 通常色 (RED..PURPLE) - これらが「自色と異なる隣接」とみなす対象
NORMAL_COLORS: frozenset[int] = frozenset({1, 2, 3, 4, 5})

# 隣接 4 セル中 N セル以上が「自色と異なる通常色」なら孤立判定
DEFAULT_MIN_DIFFERENT_NEIGHBORS: int = 3


@dataclass(frozen=True)
class IsolatedRefineResult:
    refined: Board
    n_corrected: int
    corrections: tuple[tuple[int, int, int], ...]
    # corrections: ((row, col, old_color), ...)


class IsolatedColorFilter:
    """孤立した同色 1 セルを UNKNOWN 化。"""

    def __init__(
        self,
        min_different_neighbors: int = DEFAULT_MIN_DIFFERENT_NEIGHBORS,
    ) -> None:
        self._min_diff = int(min_different_neighbors)

    def refine(self, board: Board) -> IsolatedRefineResult:
        new = board.copy()
        corrections: list[tuple[int, int, int]] = []
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                color = int(board.get(row, col))
                if color not in NORMAL_COLORS:
                    continue
                diff_neighbors = self._count_different_normal_neighbors(
                    board, row, col, color,
                )
                if diff_neighbors >= self._min_diff:
                    new.set(row, col, COLOR_UNKNOWN)
                    corrections.append((row, col, color))
        return IsolatedRefineResult(
            refined=new,
            n_corrected=len(corrections),
            corrections=tuple(corrections),
        )

    @staticmethod
    def _count_different_normal_neighbors(
        board: Board, row: int, col: int, self_color: int,
    ) -> int:
        n = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, col + dc
            if not (0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS):
                continue
            nc = int(board.get(r, c))
            if nc in NORMAL_COLORS and nc != self_color:
                n += 1
        return n


__all__ = [
    "DEFAULT_MIN_DIFFERENT_NEIGHBORS",
    "IsolatedColorFilter",
    "IsolatedRefineResult",
    "NORMAL_COLORS",
]
