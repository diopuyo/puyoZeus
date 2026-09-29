"""E20: NEXT移動までの連鎖時間と、観測設置間隔による着弾前手数。"""
from __future__ import annotations

from collections import defaultdict, deque
import math
from statistics import median
from typing import Any

import numpy as np

from src.board_state_machine import BoardState

RECENT_INTERVALS = 5
PAIR_SIZE = 2
FIRST_COLOR, LAST_COLOR = 1, 5
FINAL_PLACEMENT_HANDS = 1
# 既存の絶対終了判定と同じ連続確認数。ゲームの時間定数ではない。
NEXT_CONFIRM_FRAMES = 2


def spec_hands(remaining_animation: float, placement_interval: float) -> int:
    """確定までに置ける手数に、確定後の現在の一手を加える。"""
    if not math.isfinite(placement_interval) or placement_interval <= 0:
        raise ValueError("設置間隔は正の有限秒数が必要")
    if not math.isfinite(remaining_animation):
        raise ValueError("アニメ残時間は有限秒数が必要")
    return FINAL_PLACEMENT_HANDS + math.floor(max(0., remaining_animation) / placement_interval)


class LandingHandsObservation:
    """未来観測を使わず、同じ入力で再現できる外部の観測履歴。"""

    def __init__(self) -> None:
        self.animations: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=RECENT_INTERVALS))
        self.reset()

    def reset(self) -> None:
        """試合境界では手番をリセットする。過去に完了した演出実測は保持する。"""
        self.motion = [False, False]
        self.motion_start: list[float | None] = [None, None]
        self.queue: list[tuple | None] = [None, None]
        self.candidate: list[tuple | None] = [None, None]
        self.queue_frames = [0, 0]
        self.shifted: list[float | None] = [None, None]
        self.anchor: list[int | None] = [None, None]
        self.placed = [False, False]
        self.last_placement: list[float | None] = [None, None]
        self.intervals = [deque(maxlen=RECENT_INTERVALS), deque(maxlen=RECENT_INTERVALS)]
        self.chains: list[dict | None] = [None, None]
        self.completed: list[dict[float, float]] = [{}, {}]
        self.placements: list[list[float]] = [[], []]

    def observe(self, sides: tuple, now: float) -> None:
        """NEXT繰り上がりを裏付け、STABLE色数の二個増加を設置として観測する。"""
        for idx, side in enumerate(sides):
            self.shifted[idx] = self._next_shift(idx, side, now)
            board = getattr(side, "confirmed_board", None)
            if board is None or side.state != BoardState.STABLE:
                continue
            count = int(np.count_nonzero((board._grid >= FIRST_COLOR) & (board._grid <= LAST_COLOR)))
            if self.shifted[idx] is not None or self.anchor[idx] is None or count < self.anchor[idx]:
                self.anchor[idx], self.placed[idx] = count, False
            if count >= self.anchor[idx] + PAIR_SIZE and not self.placed[idx]:
                self._place(idx, now)

    def _next_shift(self, idx: int, side: Any, now: float) -> float | None:
        """発光だけのROI移動は棄却し、NEXT2→NEXT1の実繰り上がりを確認する。"""
        motion = bool(getattr(side, "next_slide_motion", False))
        if motion and not self.motion[idx]:
            self.motion_start[idx] = now
        self.motion[idx] = motion
        pairs = (getattr(side, "next_pair", None), getattr(side, "dnext_pair", None))
        if any(p is None or any(v < FIRST_COLOR or v > LAST_COLOR for v in p) for p in pairs):
            self.queue_frames[idx] = 0
            return None
        queue = tuple(tuple(p) for p in pairs)
        self.queue_frames[idx] = self.queue_frames[idx]+1 if queue == self.candidate[idx] else 1
        self.candidate[idx] = queue
        if self.queue_frames[idx] < NEXT_CONFIRM_FRAMES:
            return None
        previous = self.queue[idx]
        if previous is None or queue == previous:
            self.queue[idx] = queue
            if not motion:
                self.motion_start[idx] = None  # 元の色へ戻った発光は確定根拠にしない。
            return None
        shifted = queue[0] == previous[1]
        if shifted:
            self.queue[idx] = queue
        stamp = self.motion_start[idx]
        self.motion_start[idx] = None
        return (stamp if stamp is not None else now) if shifted else None

    def observe_chains(self, tracker: Any, counts: list[int], now: float) -> None:
        """段通知の仮IDを使わず、trackerが統合した連鎖の発火時刻に結び付ける。"""
        for idx, label in enumerate(("1P", "2P")):
            chain = tracker.latest_chain(label)
            if chain is None or chain.trigger_sec in self.completed[idx]:
                continue
            active = self.chains[idx]
            if active is None or active["trigger"] != chain.trigger_sec:
                active = dict(trigger=chain.trigger_sec, count=counts[idx])
                self.chains[idx] = active
                self.last_placement[idx] = None  # 演出時間は設置間隔へ混ぜない。
            active["count"] = max(active["count"], counts[idx])
            stamp = self.shifted[idx]
            if (stamp is None and getattr(chain, "end_confirmed", False)
                    and getattr(chain, "end_reason", None) in ("next", "tsumo")):
                stamp = chain.end_signal_sec  # 同色NEXT等は既存の検証済み物理終了を使う。
            if stamp is not None and chain.trigger_sec < stamp <= now:
                self.completed[idx][chain.trigger_sec] = stamp
                self.animations[active["count"]].append(stamp-chain.trigger_sec)
                self.chains[idx] = None

    def _place(self, idx: int, now: float) -> None:
        """同じNEXT手番の部分的な盤面更新を二重に数えない。"""
        if self.placed[idx]:
            return
        last = self.last_placement[idx]
        if last is not None and now > last:
            self.intervals[idx].append(now-last)
        self.last_placement[idx] = now
        self.placed[idx] = True
        self.placements[idx].append(now)

    def estimate(self, chain: Any, attacker: int, now: float, observed_count: int) -> dict:
        """未観測値を秒定数で推測せず、欠測時は最後の一手だけを保証する。"""
        receiver = 1-attacker
        intervals = list(self.intervals[receiver])
        interval = float(median(intervals)) if intervals else None
        confirmed = None if chain is None else self.completed[attacker].get(chain.trigger_sec)
        count = max(observed_count, (chain.predicted_chain_count or 0) if chain else 0)
        samples = list(self.animations[count])
        duration = float(median(samples)) if samples else None
        remaining = None if duration is None or chain is None else max(0., duration-(now-chain.trigger_sec))
        reason = "next_motion_confirmed" if confirmed is not None else "observed_animation_and_placements"
        hands = FINAL_PLACEMENT_HANDS
        if chain is None:
            reason = "no_attack_chain"
        elif confirmed is None:
            if remaining is None or interval is None:
                reason = "missing_animation" if remaining is None else "missing_placement_interval"
            else:
                hands = spec_hands(remaining, interval)
        return dict(hands=hands, reason=reason, confirmed_sec=confirmed,
                    remaining_animation_sec=0. if confirmed is not None else remaining,
                    placement_interval_sec=interval, placement_intervals=intervals,
                    animation_samples=samples, animation_chain_count=count)
