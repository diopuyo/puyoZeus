"""ライブ専用の負荷縮退。物理状態更新は間引かず、表示計算の周期だけ変える。"""
from __future__ import annotations

import os
from typing import Any

PERIODS = (0.5, 1.0, 2.0)
EXTERNAL_LEVELS = (1.0, 2.0)
LAG_LEVELS = (0.10, 0.25)
RECOVER_SECONDS = 5.0
EVENT_TAIL_SEC = 4/30
EVENT_BACKLOG_SLOTS = 2


class EvaluationPacer:
    def __init__(self, realtime: bool) -> None:
        self.enabled = realtime and os.environ.get('PUYO_ADAPTIVE_EVALUATION') == '1'
        self.sampler: Any = None
        self.next_sample = 0.0
        self.level, self.recover_at = 0, None
        self.last: dict = {}

    def period(self, now: float, lag: float) -> float:
        if not self.enabled or now < self.next_sample:
            return PERIODS[self.level]
        from .live_load import CpuSampler, SAMPLE_SECONDS
        self.next_sample = now+SAMPLE_SECONDS
        if self.sampler is None and os.name == 'posix':
            self.sampler = CpuSampler()
        elif self.sampler is not None:
            self.last = self.sampler.sample()
        external = self.last.get('external_cores', 0) if self.last.get('valid') else 0
        target = max(sum(external > value for value in EXTERNAL_LEVELS),
                     sum(lag > value for value in LAG_LEVELS))
        self.update(target, now)
        return PERIODS[self.level]

    def update(self, target: int, now: float) -> None:
        if target >= self.level:
            self.level, self.recover_at = target, None
        elif self.recover_at is None:
            self.recover_at = now+RECOVER_SECONDS
        elif now >= self.recover_at:
            self.level, self.recover_at = self.level-1, None


class EventPriority:
    """直前の置き/連鎖/着弾兆候の後だけ連続観測を優先し、滞留は2slotに制限する。"""
    def __init__(self) -> None:
        self.enabled = os.environ.get('PUYO_EVENT_PRIORITY') == '1'
        self.until = float('-inf')
        self.counts: tuple | None = None

    def observe(self, notice: Any) -> None:
        if not self.enabled:
            return
        result = notice.result()
        critical = self.counts != notice.pipeline.counts
        for side in (result.p1, result.p2):
            state = getattr(side.state, 'name', str(side.state))
            critical |= state != 'STABLE' or bool(side.chain_event) or bool(side.landing_chain_started)
        self.counts = notice.pipeline.counts
        if critical:
            self.until = notice.t_sec+EVENT_TAIL_SEC

    def select(self, next_index: int, due: int, stride: int, fps: float) -> int:
        if self.enabled and next_index/fps <= self.until:
            return max(next_index, due-EVENT_BACKLOG_SLOTS*stride)
        return due


def feedback_event(source: Any, notice: Any) -> None:
    while hasattr(source, 'source'):
        source = source.source
    priority = getattr(source, 'event_priority', None)
    if priority is not None:
        priority.observe(notice)
