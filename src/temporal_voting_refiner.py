"""W11-D: 過去 N フレームの cell-level 多数決で出力平滑化。

StatefulBoardTracker は accept-then-lock 型 (一度確定したら保持) なので、
本レイヤーは raw CNN output (tracker 前) を平滑化する目的。

戦略:
    1. 過去 N フレームの (side, row, col) → 色 を保持
    2. 現フレーム判定: 過去 N の最頻色を採用
    3. 過去 N が EM 多数 + 現フレーム色 → 「初出現」と判断、現フレーム採用
    4. 過去 N が色多数 + 現フレーム EM → 「消滅扱い」だが score 変化なければ
       色を維持 (CNN の momentary EM 誤検出の救済)

これは BG-EM/ScorePhysics より安全:
    - 多数決ベースなので 1 フレームのノイズに耐性
    - CNN を尊重しつつ瞬間的な揺れを除去
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from src.board import (
    BOARD_COLS, COLOR_EMPTY, HIDDEN_ROWS, Board,
)


DEFAULT_WINDOW: int = 3


@dataclass
class TemporalVotingRefiner:
    """過去 N フレームの cell-level 多数決による平滑化。"""

    window: int = DEFAULT_WINDOW
    # (side, vrow, col) -> deque of int (色コード)
    history: dict[tuple[str, int, int], deque[int]] = field(
        default_factory=dict,
    )
    # 現フレームの色を全面採用するか、最頻色か
    # vote_majority: 最頻色を採用 (推奨)
    # current_priority: 現フレームを優先 (window 全部一致のときのみ override)
    mode: str = "vote_majority"

    def reset(self) -> None:
        self.history = {}

    def _get_deque(self, key: tuple[str, int, int]) -> "deque[int]":
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window)
        return self.history[key]

    def refine(self, side: str, board: Board) -> Board:
        out = board.copy()
        for vrow in range(12):
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                key = (side, vrow, col)
                cur = int(out.get(row, col))
                dq = self._get_deque(key)
                dq.append(cur)
                if len(dq) < self.window:
                    continue
                if self.mode == "current_priority":
                    # window 全部一致なら override (極端な変化を防ぐ)
                    if len(set(dq)) == 1:
                        continue  # 全部同じ = 安定
                    # 不安定なら最頻色採用
                counts: dict[int, int] = {}
                for c in dq:
                    counts[c] = counts.get(c, 0) + 1
                vote_color, vote_count = max(
                    counts.items(), key=lambda kv: kv[1],
                )
                # 過半数 (window=3 なら 2/3 以上) を要求
                if vote_count * 2 > self.window:
                    if vote_color != cur:
                        out.set(row, col, vote_color)
        return out


__all__ = ["DEFAULT_WINDOW", "TemporalVotingRefiner"]
