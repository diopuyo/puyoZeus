"""W10-B: score 連動の物理推論補正。

「score 変化なし & 過去 N フレーム連続 EM」だったセルが現フレームで色付き
判定された場合、それは CNN の hallucination の可能性が高いので EM 維持。

逆に score が増えた場合 (chain 発火 or pair 着地) は cell 変化を許容。

戦略:
    - score_delta = score_t - score_{t-1}
    - 各セルごとに em_streak (連続 EM フレーム数) を保持
    - em_streak >= MIN_EM_STREAK AND score_delta < SCORE_DELTA_THRESHOLD AND
      current_frame で色付き → EM に強制
    - em_streak >= MIN_EM_STREAK AND score 増加 → 色付きを許容 (em_streak リセット)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA,
    COLOR_UNKNOWN, HIDDEN_ROWS, Board,
)


# 最小 EM streak (このフレーム数連続 EM だった cell は信頼)
MIN_EM_STREAK: int = 3
# score がこれ以下の変化なら "chain なし" とみなす
SCORE_DELTA_THRESHOLD: int = 30


@dataclass
class ScorePhysicsRefiner:
    """score + 多フレーム EM 整合の物理推論補正。"""

    min_em_streak: int = MIN_EM_STREAK
    score_delta_threshold: int = SCORE_DELTA_THRESHOLD

    # 各セルごとの em_streak: (side, row, col) → int
    em_streaks: dict[tuple[str, int, int], int] = field(default_factory=dict)
    prev_score: dict[str, int | None] = field(
        default_factory=lambda: {"1P": None, "2P": None}
    )

    # 統計用
    n_em_overrides: int = 0
    n_color_allowed: int = 0

    def reset(self) -> None:
        self.em_streaks = {}
        self.prev_score = {"1P": None, "2P": None}
        self.n_em_overrides = 0
        self.n_color_allowed = 0

    def refine(
        self, side: str, board: Board, current_score: int | None,
    ) -> Board:
        """1 side の board を refine。

        Args:
            side: "1P" or "2P"
            board: CNN の予測結果 (BOARD_ROWS × BOARD_COLS)
            current_score: ScoreOcr 結果 (None なら不確定 → 補正スキップ)

        Returns:
            refine 済み board (新しい Board オブジェクト)
        """
        out = board.copy()
        prev = self.prev_score[side]

        # score 変化判定
        score_delta = 0
        score_known = False
        if current_score is not None and prev is not None:
            score_delta = int(current_score - prev)
            score_known = True

        chain_fired = score_known and score_delta > self.score_delta_threshold

        # 全セル更新
        for vrow in range(12):  # visible rows
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                key = (side, vrow, col)
                streak = self.em_streaks.get(key, 0)
                color = int(out.get(row, col))

                if color == COLOR_EMPTY:
                    self.em_streaks[key] = streak + 1
                elif color == COLOR_UNKNOWN:
                    # UNKNOWN は streak 維持 (reset しない)
                    pass
                else:
                    # 色付き
                    if (
                        streak >= self.min_em_streak
                        and not chain_fired
                        and color != COLOR_OJAMA
                    ):
                        # 連続 EM だった cell に色付き判定 → hallucination
                        # (OJAMA は降ってくるので例外、強制しない)
                        out.set(row, col, COLOR_EMPTY)
                        self.em_streaks[key] = streak + 1
                        self.n_em_overrides += 1
                    else:
                        # 色付き許容、streak リセット
                        self.em_streaks[key] = 0
                        if streak >= self.min_em_streak:
                            self.n_color_allowed += 1

        if current_score is not None:
            self.prev_score[side] = current_score
        return out

    def get_em_streaks_summary(self) -> dict[int, int]:
        """各 streak 値ごとの cell 数 (debug 用)。"""
        out: dict[int, int] = {}
        for s in self.em_streaks.values():
            out[s] = out.get(s, 0) + 1
        return out


__all__ = [
    "MIN_EM_STREAK",
    "SCORE_DELTA_THRESHOLD",
    "ScorePhysicsRefiner",
]
