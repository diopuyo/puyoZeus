"""観測した落下区間に限って配送量を数える。予告だけでは着弾を発生させない。"""
from __future__ import annotations

from dataclasses import dataclass

from src.scoring import OJAMA_MAX_DROP_PER_TURN

# 状態のちらつきを同じ落下としてまとめる。既存B3着弾区間の終了待ちと同じ秒数。
LANDING_CUE_CLOSE_SEC: float = 1.0


@dataclass
class LandingObservation:
    """一つの観測落下区間（状態表示または大きな持続増分）。同じ合図で重複控除しない。"""

    last_fall_t: float | None = None
    pending_before: int = 0
    consumed: int = 0
    increase: int = 0
    attempted: int = 0

    def observe(self, falling: bool, t: float, pending: int) -> None:
        """着弾の合図を観測した時点の予告を固定し、後から来る攻撃を予算に混ぜない。"""
        if self.last_fall_t is not None and t - self.last_fall_t > LANDING_CUE_CLOSE_SEC:
            self.last_fall_t = None
        if not falling:
            return
        if self.last_fall_t is None:
            self.pending_before = pending
            self.consumed = self.increase = self.attempted = 0
        self.last_fall_t = t

    def amount(self, increase: int) -> int:
        """同じ区間の累積増分と配送上限の大きい方から、既に依頼した量を引く。"""
        if self.last_fall_t is None:
            return 0
        self.increase += increase
        target = max(min(self.pending_before, OJAMA_MAX_DROP_PER_TURN), self.increase)
        amount = max(0, target - self.attempted)
        # 台帳不足で差し引けなくても、後で来る攻撃から穴埋めしない。
        self.attempted = target
        return amount
