"""W6: 4+ 同色連結 (物理矛盾) を検出して補正する。

ぷよぷよでは 4 個以上の同色連結が接地した瞬間に連鎖が始まる。
単一安定フレームでこの状態が観察される = CNN 認識誤りの可能性大
(連鎖アニメ中の中途半端なフレームは呼び出し側で除外する前提)。

補正方針:
    各 4+ 連結グループから「最も外周のセル」(同色隣接が最少のセル) を
    1 つ選び、COLOR_UNKNOWN に変更。これで連結が 3 以下になり、連鎖発火
    の物理矛盾が解消される。確率的盤面では UNKNOWN セルは複数色の分布で
    後段が補完可能。

使い方:
    refiner = PhysicalSanityRefiner()
    refined, result = refiner.refine(board)
    if result.n_corrected > 0:
        # CNN 認識ミス補正済
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
from src.chain import ChainSimulator, MIN_ERASE_COUNT


@dataclass(frozen=True)
class SanityRefineResult:
    """補正結果。"""
    refined: Board
    n_corrected: int
    corrections: tuple[tuple[int, int, int, int], ...]
    # corrections: ((row, col, old_color, new_color), ...)


class PhysicalSanityRefiner:
    """4+ 同色連結を検出して 1 セルを UNKNOWN に置換。"""

    def __init__(self, simulator: ChainSimulator | None = None) -> None:
        self._simulator = simulator or ChainSimulator()

    def refine(self, board: Board) -> SanityRefineResult:
        """4+ 同色連結グループから「最外周セル」を 1 つ UNKNOWN 化。

        OJAMA は消えないので除外。
        """
        new = board.copy()
        corrections: list[tuple[int, int, int, int]] = []
        groups = self._simulator.find_groups(board)
        for group in groups:
            if group.color == COLOR_OJAMA:
                continue
            if group.color == COLOR_EMPTY:
                continue
            if group.size < MIN_ERASE_COUNT:
                continue
            target = self._pick_most_outer(group)
            if target is None:
                continue
            row, col = target
            corrections.append((row, col, int(group.color), COLOR_UNKNOWN))
            new.set(row, col, COLOR_UNKNOWN)
        return SanityRefineResult(
            refined=new,
            n_corrected=len(corrections),
            corrections=tuple(corrections),
        )

    @staticmethod
    def _pick_most_outer(group) -> tuple[int, int] | None:
        """グループ内で「同色隣接数が最少」のセルを返す (= 最外周)。

        同点の場合、最上段のセルを選ぶ (連鎖発火点になりやすい位置を優先補正)。
        """
        cells_set = set(group.cells)
        best: tuple[int, int] | None = None
        best_score: tuple[int, int, int] = (999, 999, 999)
        for r, c in group.cells:
            same_neighbors = 0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells_set:
                    same_neighbors += 1
            score = (same_neighbors, r, c)
            if score < best_score:
                best_score = score
                best = (r, c)
        return best


__all__ = [
    "PhysicalSanityRefiner",
    "SanityRefineResult",
]
