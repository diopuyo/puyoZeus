"""V2.3: 連結同色グループ多数色による異色 1 セル補正。

ぷよぷよでは色は局所的に同色クラスタを形成しやすい。1 セルだけ周囲と
異なる色が CNN/HSV で検出された場合、それは認識ミスの可能性が高い。
本モジュールは隣接 4 セルの多数色で「孤立した異色 1 セル」のみ補正する。

設計上の注意 (誤補正回避):
    - 自色が EMPTY/UNKNOWN/OJAMA は補正対象外 (色なし or 別系統)
    - 隣接の同色多数決閾値は厳しめ (デフォルト 3/4 = 75%)
    - 補正は 1 回のみ (連鎖伝搬しない)
    - 「赤赤赤 / 紫赤赤」のような正しい配置 (紫が境界) を壊さないため、
      隣接 4 セル中 3 セル以上が同色 (= 自分が完全に囲まれてる) を条件に
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    Board,
)

# 補正対象外の色 (色なし系)
EXCLUDE_COLORS: frozenset[int] = frozenset({
    COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA,
})

# デフォルト: 隣接 4 セル中 何セル以上が同色なら補正するか
DEFAULT_MIN_NEIGHBOR_AGREEMENT: int = 3


@dataclass(frozen=True)
class RefineResult:
    """補正結果。"""
    refined: Board
    n_corrected: int
    corrections: tuple[tuple[int, int, int, int], ...]
    # corrections: ((row, col, old_color, new_color), ...)


class ConnectivityShapeRefiner:
    """連結グループ整合性に基づく色補正。"""

    def __init__(
        self,
        min_neighbor_agreement: int = DEFAULT_MIN_NEIGHBOR_AGREEMENT,
    ) -> None:
        self._min_agreement = int(min_neighbor_agreement)

    def refine(self, board: Board) -> RefineResult:
        """孤立した異色 1 セルを多数色で補正。

        補正条件: あるセル (row, col) について
            - 自色は通常色 (EMPTY/UNKNOWN/OJAMA でない)
            - 隣接 4 セル (上下左右) のうち、通常色のセルを集計
            - 多数色 c が min_neighbor_agreement 以上、かつ c != 自色

        Returns:
            RefineResult: 補正後盤面と統計。
        """
        new = board.copy()
        corrections: list[tuple[int, int, int, int]] = []
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                color = int(board.get(row, col))
                if color in EXCLUDE_COLORS:
                    continue
                neighbors = self._neighbor_colors(board, row, col)
                if len(neighbors) < self._min_agreement:
                    continue
                counter = Counter(neighbors)
                most_color, most_count = counter.most_common(1)[0]
                if most_color in EXCLUDE_COLORS:
                    continue
                if most_color == color:
                    continue
                if most_count < self._min_agreement:
                    continue
                # 補正
                new.set(row, col, most_color)
                corrections.append((row, col, color, most_color))
        return RefineResult(
            refined=new,
            n_corrected=len(corrections),
            corrections=tuple(corrections),
        )

    @staticmethod
    def _neighbor_colors(board: Board, row: int, col: int) -> list[int]:
        """指定セルの上下左右の通常色 (EXCLUDE_COLORS 除外) を返す。"""
        out: list[int] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, col + dc
            if not (0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS):
                continue
            nc = int(board.get(r, c))
            if nc in EXCLUDE_COLORS:
                continue
            out.append(nc)
        return out


__all__ = [
    "ConnectivityShapeRefiner",
    "DEFAULT_MIN_NEIGHBOR_AGREEMENT",
    "EXCLUDE_COLORS",
    "RefineResult",
]
