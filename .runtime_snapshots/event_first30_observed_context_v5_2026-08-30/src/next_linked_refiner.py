"""V2.1: ネクストペア連動による盤面新規出現ぷよの色補正。

t-1 のネクストペア (NextDetector の出力) は、t で盤面に出現するぷよの色を
論理的に確定的に決める。CNN/HSV 認識誤差により色がズレた場合、
ネクストペアの色集合に補正することで認識精度を底上げできる。

設計方針:
    - 安定フレーム (落下完了後) のみで動作。連鎖中・落下中フレームは
      呼び出し側 (AnimationFilter) で除外する前提。
    - 「新規出現セル」= prev EMPTY → cur 非 EMPTY のセル。
    - 期待値: 新規出現セル数 == 2 (ペア)。それ以外は補正対象外。
    - ペア色集合 (順序無視 multiset) と next_pair の色集合を比較。
      不一致なら割り当て補正 (Hungarian 風の最小コスト割当て)。

Edge cases:
    - next_pair に COLOR_EMPTY/UNKNOWN が含まれる → 補正スキップ
    - 新規出現が 1 セル / 3 セル以上 → 補正スキップ (PairAppearanceConsistency に委ねる)
    - 連鎖中の落下完了は新規出現ではなく既存ぷよの再配置 → 呼び出し側で除外
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    HIDDEN_ROWS,
    Board,
)


# next_pair に含まれていたら補正をスキップする色 (確定的でない)
SKIP_COLORS: frozenset[int] = frozenset({
    COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA,
})


@dataclass(frozen=True)
class RefineResult:
    """補正結果。"""
    refined: Board
    n_new_cells: int
    n_corrected: int
    skipped_reason: str | None = None


def _collect_new_cells(
    prev: Board, cur: Board,
) -> list[tuple[int, int, int]]:
    """prev EMPTY → cur 非 EMPTY のセルを (row, col, cur_color) で返す。

    隠し段 (row < HIDDEN_ROWS) は除外 (UNKNOWN が混じるため)。
    """
    out: list[tuple[int, int, int]] = []
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            p = int(prev.get(row, col))
            c = int(cur.get(row, col))
            if p == COLOR_EMPTY and c != COLOR_EMPTY and c != COLOR_UNKNOWN:
                out.append((row, col, c))
    return out


def _multiset_equal(a: Iterable[int], b: Iterable[int]) -> bool:
    return sorted(a) == sorted(b)


def _assign_colors_to_cells(
    cells: list[tuple[int, int, int]],
    target_colors: list[int],
) -> list[int]:
    """各セルに target_colors を割り当てる。

    cells[i] の現在色が target_colors にあれば優先採用、
    残りは最も近い (= 単純に余った) 色を割り当てる。

    Returns:
        各 cell の補正後色リスト (cells と同じ順序)。
    """
    assert len(cells) == len(target_colors)
    n = len(cells)
    used = [False] * n
    out = [0] * n
    # まず既に target に含まれる cell を優先割当
    remaining_targets = list(target_colors)
    for i, (_, _, cur_c) in enumerate(cells):
        if cur_c in remaining_targets:
            remaining_targets.remove(cur_c)
            out[i] = cur_c
            used[i] = True
    # 残りに余った target を順次割り当て
    j = 0
    for i in range(n):
        if used[i]:
            continue
        out[i] = remaining_targets[j]
        j += 1
    return out


class NextLinkedColorRefiner:
    """ネクストペア連動による色補正。"""

    def refine(
        self,
        prev_board: Board,
        cur_board: Board,
        prev_next_pair: tuple[int, int],
    ) -> RefineResult:
        """prev_next_pair に基づいて cur_board の新規 2 セル色を補正。

        Args:
            prev_board: t-1 の盤面 (既に補正済の安定状態が望ましい)
            cur_board: t の盤面 (CNN 認識直後)
            prev_next_pair: (top_color, bot_color) — t-1 時点で観測された
                            「次に降ってくるはずのペア」

        Returns:
            RefineResult: 補正後盤面と統計。
        """
        # 確定的でない色を含むなら補正スキップ
        if any(c in SKIP_COLORS for c in prev_next_pair):
            return RefineResult(
                refined=cur_board.copy(),
                n_new_cells=0,
                n_corrected=0,
                skipped_reason="next_pair_not_definite",
            )

        new_cells = _collect_new_cells(prev_board, cur_board)
        if len(new_cells) != 2:
            return RefineResult(
                refined=cur_board.copy(),
                n_new_cells=len(new_cells),
                n_corrected=0,
                skipped_reason=f"new_cell_count={len(new_cells)}",
            )

        target = list(prev_next_pair)
        cur_colors = [c for _, _, c in new_cells]

        # 色集合一致なら補正不要
        if _multiset_equal(cur_colors, target):
            return RefineResult(
                refined=cur_board.copy(),
                n_new_cells=2,
                n_corrected=0,
            )

        # 不一致 → 色割当て補正
        assigned = _assign_colors_to_cells(new_cells, target)
        refined = cur_board.copy()
        n_corrected = 0
        for (row, col, _), new_c in zip(new_cells, assigned):
            if int(refined.get(row, col)) != new_c:
                refined.set(row, col, new_c)
                n_corrected += 1
        return RefineResult(
            refined=refined,
            n_new_cells=2,
            n_corrected=n_corrected,
        )


__all__ = [
    "NextLinkedColorRefiner",
    "RefineResult",
    "SKIP_COLORS",
]
