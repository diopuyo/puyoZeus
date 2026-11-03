"""個数レビュー専用の相殺前送り量。得点の端数とマージンを段ごとに保つ。"""
from __future__ import annotations

from src.phase_j._live_ex_frozen.src.scoring import compute_effective_rate, ALL_CLEAR_BONUS

STEP_SEC = 1.4  # 既存score_ocrの段周期観測を継承。動画80では調整しない。
FORMULA_SCORE_DELAY_SEC = .9  # 既存score_ocrの約0.933秒表示の30fps検出差を含む近似。
CONFIRM_SAMPLES = 2


def gross(scores: list[int], times: list[float], carry: int,
          first_move: float, all_clear: bool = False) -> tuple[int, int]:
    """落下点は発火前carryへ、次連鎖に持越した全消しは初段へ加える。"""
    if len(scores) != len(times) or carry < 0:
        raise ValueError('得点・時刻・端数が不整合')
    sent = 0
    for i, (score, stamp) in enumerate(zip(scores, times)):
        rate = compute_effective_rate(stamp-first_move, from_first_move=True)
        pool = carry+score+(ALL_CLEAR_BONUS if all_clear and i == 0 else 0)
        amount, carry = divmod(pool, rate)
        sent += amount
    return sent, carry


class ScorePrefix:
    """配送済み通常得点だけを処理する。初手未確定時は送り量を欠測にする。"""

    def __init__(self, first_move_quiet_sec: float | None = None) -> None:
        self.score = [None, None]
        self.carry = [0, 0]
        self.pending = [None, None]
        self.run = [0, 0]
        self.history_complete = [False, False]
        self.first_move: float | None = None
        self.latest = float('-inf')
        self.quiet_sec = first_move_quiet_sec
        self.changed_at: list[float | None] = [None, None]

    def observe(self, side: int, stamp: float, score: int | None) -> None:
        if stamp < self.latest:
            raise ValueError('得点通知は時刻順で渡してください')
        self.latest = stamp
        if score is None:
            self.run[side] = 0
            return
        self.run[side] = self.run[side]+1 if self.pending[side] == score else 1
        if self.pending[side] != score and score > 0:
            self.changed_at[side] = stamp
        self.pending[side] = score
        changed = self.changed_at[side]
        if (self.first_move is None and self.quiet_sec is not None and changed is not None
                and self.history_complete[side] and stamp-changed >= self.quiet_sec):
            self.first_move = changed
        if self.run[side] < CONFIRM_SAMPLES:
            return
        previous = self.score[side]
        if score == 0 and not self.history_complete[side]:
            self.history_complete[side] = True
            self.score[side], self.carry[side] = 0, 0
            return
        if previous is not None and score < previous:
            return
        if previous is not None and score > previous:
            if self.history_complete[side]:
                _, self.carry[side] = gross([score-previous], [stamp], self.carry[side],
                                           self.first_move if self.first_move is not None else stamp)
        self.score[side] = score

    def snapshot(self, side: int, stamp: float) -> dict:
        if self.latest > stamp:
            raise ValueError('未来の得点台帳')
        return dict(carry=self.carry[side] if self.history_complete[side] else None, score=self.score[side],
                    history_complete=self.history_complete[side],
                    first_move=self.first_move, observed_sec=self.latest)
