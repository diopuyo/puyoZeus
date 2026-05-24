"""V2.2: ぷよペア出現の整合性チェック。

ぷよぷよではぷよが必ずペア (2 個) で落下する。1 個または 3 個以上が
新規出現するのは認識ミスの兆候 (片方が EMPTY と誤認 / 既存ぷよが急に
色変わったように誤認)。

本モジュールは「新規 1 セル出現の場合の相方位置候補」を提示する補助
ユーティリティ。実際の補正は呼び出し側 (V2.4 StatefulBoardTracker 拡張)
が判断する。

設計思想:
    - V2.1 NextLinkedColorRefiner: 2 セル出現で色不一致 → 色補正
    - V2.2 PairAppearanceConsistency: 1 セル出現 → 相方位置推定 (本モジュール)
    - V2.3 ConnectivityShapeRefiner: 静的盤面の異色 1 セル補正
    - V2.4 StatefulBoardTracker: 上記 3 つを統合した時系列フィルタ

ぷよペアの落下後配置パターン:
    縦置き: 同列で連続 2 セル (row, col) + (row-1, col)
    横置き: 同行で隣接 2 セル (row, col) + (row, col±1)
"""
from __future__ import annotations

from dataclasses import dataclass

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_UNKNOWN,
    HIDDEN_ROWS,
    Board,
)


@dataclass(frozen=True)
class PairConsistencyResult:
    """ペア整合性チェック結果。"""
    n_new_cells: int
    new_cells: tuple[tuple[int, int, int], ...]  # (row, col, color)
    is_consistent: bool  # 新規 0 or 2 セルなら True
    candidate_partner_positions: tuple[tuple[int, int], ...]
    # 1 セル新規出現時の「相方位置候補」(row, col)。0/2 セルなら空


class PairAppearanceConsistency:
    """ぷよペア出現の整合性をチェック。"""

    def check(
        self, prev_board: Board, cur_board: Board,
    ) -> PairConsistencyResult:
        """prev → cur の新規出現セルを集計し、ペア整合性を評価。

        Args:
            prev_board: t-1 の盤面
            cur_board: t の盤面

        Returns:
            PairConsistencyResult: 統計と相方位置候補。
        """
        new_cells = self._collect_new_cells(prev_board, cur_board)
        if len(new_cells) in (0, 2):
            return PairConsistencyResult(
                n_new_cells=len(new_cells),
                new_cells=tuple(new_cells),
                is_consistent=True,
                candidate_partner_positions=(),
            )

        # 1 セル出現のときのみ相方位置候補を返す
        if len(new_cells) == 1:
            row, col, _ = new_cells[0]
            candidates = self._partner_position_candidates(
                row, col, prev_board, cur_board,
            )
            return PairConsistencyResult(
                n_new_cells=1,
                new_cells=tuple(new_cells),
                is_consistent=False,
                candidate_partner_positions=tuple(candidates),
            )

        # 3 セル以上 → 異常
        return PairConsistencyResult(
            n_new_cells=len(new_cells),
            new_cells=tuple(new_cells),
            is_consistent=False,
            candidate_partner_positions=(),
        )

    @staticmethod
    def _collect_new_cells(
        prev: Board, cur: Board,
    ) -> list[tuple[int, int, int]]:
        """prev EMPTY → cur 非 EMPTY (UNKNOWN 以外) のセル一覧。"""
        out: list[tuple[int, int, int]] = []
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                p = int(prev.get(row, col))
                c = int(cur.get(row, col))
                if p == COLOR_EMPTY and c not in (COLOR_EMPTY, COLOR_UNKNOWN):
                    out.append((row, col, c))
        return out

    @staticmethod
    def _partner_position_candidates(
        row: int, col: int, prev: Board, cur: Board,
    ) -> list[tuple[int, int]]:
        """1 セル新規出現位置 (row, col) の相方候補位置を返す。

        候補:
            - (row-1, col): 縦置きの上ぷよ
            - (row, col-1) / (row, col+1): 横置きの隣
        条件: 候補位置が prev でも cur でも EMPTY (= 認識ミスの可能性)
        """
        candidates: list[tuple[int, int]] = []
        for dr, dc in ((-1, 0), (0, -1), (0, 1)):
            r, c = row + dr, col + dc
            if not (HIDDEN_ROWS <= r < BOARD_ROWS and 0 <= c < BOARD_COLS):
                continue
            if (
                int(prev.get(r, c)) == COLOR_EMPTY
                and int(cur.get(r, c)) == COLOR_EMPTY
            ):
                candidates.append((r, c))
        return candidates


__all__ = [
    "PairAppearanceConsistency",
    "PairConsistencyResult",
]
