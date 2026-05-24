"""W13-A: 新着 cell 色を active next_pair に拘束。

ロジック:
    cell が前フレーム EM → 現フレーム COLOR の遷移を検出。その COLOR が以下に
    含まれていなければ suspicious とみなし UNKNOWN 化:
        - 現フレームの next_pair の色
        - 現フレームの dnext_pair の色
        - 前フレームの next_pair の色 (1 ペア前のツモが着地済の可能性)

これにより、CNN が突然 hallucination で生成する孤立色 cell を抑制。
動画ごとの color profile に依らない物理拘束なので robust。

Note:
    OJAMA は降下時に新着するので例外 (next_pair に含まれない場合も許容)。
    EnhancedBoardTracker (V2.1+) と機能重複するが、こちらは現フレーム単位の
    EM→COLOR 検知なので独立して運用可能。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.board import (
    BOARD_COLS, COLOR_EMPTY, COLOR_OJAMA,
    COLOR_UNKNOWN, HIDDEN_ROWS, Board,
)


@dataclass
class PairLandingCheck:
    """新着 cell 色が next_pair に含まれない場合は UNKNOWN 化。"""

    # ロールバック対象から除外する色 (常に許容)
    allowed_extra_colors: set[int] = field(
        default_factory=lambda: {COLOR_OJAMA, COLOR_EMPTY, COLOR_UNKNOWN}
    )
    prev_board: dict[str, Board | None] = field(
        default_factory=lambda: {"1P": None, "2P": None}
    )
    prev_next: dict[str, tuple[int, int] | None] = field(
        default_factory=lambda: {"1P": None, "2P": None}
    )
    prev_dnext: dict[str, tuple[int, int] | None] = field(
        default_factory=lambda: {"1P": None, "2P": None}
    )
    n_overrides: int = 0

    def reset(self) -> None:
        self.prev_board = {"1P": None, "2P": None}
        self.prev_next = {"1P": None, "2P": None}
        self.prev_dnext = {"1P": None, "2P": None}
        self.n_overrides = 0

    def refine(
        self,
        side: str,
        board: Board,
        cur_next: tuple[int, int] | None,
        cur_dnext: tuple[int, int] | None,
    ) -> Board:
        """1 side の board を refine。

        CRITICAL: next_pair が None (検出失敗) のときは制約を適用しない。
        前回バグ: 検出失敗時に全色が allowed に入らず、真のぷよが UNKNOWN 化されていた。
        """
        out = board.copy()
        prev = self.prev_board.get(side)
        prev_next = self.prev_next.get(side)
        # next_pair が全 None なら制約適用を skip (情報不足)
        any_pair_known = (
            cur_next is not None or cur_dnext is not None
            or prev_next is not None
        )
        # 新着 cell 検査は前フレームが必要 + next_pair が分かっていること
        if prev is not None and any_pair_known:
            allowed: set[int] = set(self.allowed_extra_colors)
            for pair in (cur_next, cur_dnext, prev_next):
                if pair is not None:
                    allowed.add(int(pair[0]))
                    allowed.add(int(pair[1]))
            for vrow in range(12):
                row = vrow + HIDDEN_ROWS
                for col in range(BOARD_COLS):
                    cur_color = int(out.get(row, col))
                    prev_color = int(prev.get(row, col))
                    if (
                        prev_color == COLOR_EMPTY
                        and cur_color not in allowed
                        and cur_color not in (
                            COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN,
                        )
                    ):
                        # 同色の neighbor があれば既存クラスタの一部 → 許容
                        has_same_neighbor = False
                        for dr, dc in (
                            (-1, 0), (1, 0), (0, -1), (0, 1),
                        ):
                            nr, nc = row + dr, col + dc
                            if (
                                HIDDEN_ROWS <= nr < HIDDEN_ROWS + 12
                                and 0 <= nc < BOARD_COLS
                            ):
                                if int(out.get(nr, nc)) == cur_color:
                                    has_same_neighbor = True
                                    break
                        if has_same_neighbor:
                            continue
                        # 孤立した新着色 + next_pair 不一致 → hallucination
                        out.set(row, col, COLOR_UNKNOWN)
                        self.n_overrides += 1
        self.prev_board[side] = board.copy()
        self.prev_next[side] = cur_next
        self.prev_dnext[side] = cur_dnext
        return out


__all__ = ["PairLandingCheck"]
