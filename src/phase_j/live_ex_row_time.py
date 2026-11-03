"""行の入力時刻までの鍵と保持を使う。観測の先行で過去行を失効させない。"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import logging
from typing import Any

from .live_ex_fresh import validate_ex_fresh_sec
from .live_ex_hold import LiveExHold, FRESH_SEC

MAX_KEYS = 65536
MAX_RESULTS = 8192
MAX_ROWS = 8192
LOGGER = logging.getLogger(__name__)


class RowTimeEx:
    def __init__(self, fresh_sec: float = FRESH_SEC) -> None:
        self.fresh_sec = validate_ex_fresh_sec(fresh_sec)
        self.keys: deque[dict] = deque()
        self.rows: dict[tuple, LiveExHold] = {}
        self.floor = -float('inf')
        self.epoch: Any = None
        self.pruned = 0
        self.candidates = 0
        self.applied: list[dict] = []
        self.overflow_keys = 0
        self.overflow_results = 0
        self.overflow_rows = 0
        self._overflow_warned = False

    def _record_overflow(self, kind: str, count: int) -> None:
        """破棄数はインスタンスの生存期間で累積し、警告は一度だけ出す。"""
        if not count:
            return
        name = 'overflow_' + kind
        setattr(self, name, getattr(self, name) + count)
        if not self._overflow_warned:
            LOGGER.warning('EXの履歴が上限を超えたため古い履歴を破棄します (%s: %d件)', kind, count)
            self._overflow_warned = True

    def _limit_rows(self) -> None:
        excess = max(0, len(self.rows) - MAX_ROWS)
        if excess:
            for identity in sorted(self.rows, key=lambda i: i[1])[:excess]:
                del self.rows[identity]
            self._record_overflow('rows', excess)

    def boundary(self, epoch: Any) -> None:
        if epoch != self.epoch:
            self.keys.clear()
            self.rows.clear()
            self.floor = -float('inf')
            self.epoch = epoch

    def observe(self, key: dict) -> None:
        if key['epoch'] != self.epoch:
            return
        self.keys.append(key)
        excess = max(0, len(self.keys) - MAX_KEYS)
        for _ in range(excess):
            self.keys.popleft()
        self._record_overflow('keys', excess)

    def key_at(self, t: float) -> dict:
        return next((k for k in reversed(self.keys) if k['input_t'] <= t), {})

    def prune(self, completed: deque[dict]) -> deque[dict]:
        kept = deque(r for r in completed if r['epoch'] == self.epoch
                     and r['input_t'] >= self.floor - self.fresh_sec)
        self.pruned += len(completed) - len(kept)
        # floor 直前の鍵と保持を一つ残す。解除の取りこぼしを防ぐ。
        while len(self.keys) > 1 and self.keys[1]['input_t'] <= self.floor:
            self.keys.popleft()
        older = [i for i in self.rows if i[1] < self.floor]
        for identity in sorted(older, key=lambda i: i[1])[:-1]:
            del self.rows[identity]
        excess = max(0, len(kept) - MAX_RESULTS)
        if excess:
            oldest = sorted(enumerate(kept), key=lambda pair: pair[1]['input_t'])[:excess]
            discarded = {i for i, _ in oldest}
            kept = deque(r for i, r in enumerate(kept) if i not in discarded)
            self._record_overflow('results', excess)
        self._limit_rows()
        return kept

    def select(self, completed: deque[dict], now: float, t: float,
               cutoff: float, error: str | None) -> tuple[dict | None, str]:
        key = self.key_at(t)
        self.candidates = 0
        if error:
            return None, 'worker_failed'
        if not key:
            return None, 'not_completed'
        if not key['fight']:
            return None, 'outside_exchange'
        candidates = [r for r in completed if r['epoch'] == self.epoch
                      and r['key'] == key['key'] and r['fight']
                      and 0 <= t-r['input_t'] <= self.fresh_sec
                      and r['received_at'] <= now and r['eligible_at'] <= cutoff]
        self.candidates = len(candidates)
        if not candidates:
            return None, 'not_ready_or_stale'
        return max(candidates, key=lambda r: r['serial']), 'available'

    def held(self, now: float, t: float, cutoff: float, source: dict | None,
             reason: str) -> tuple[dict | None, str]:
        identity = (self.epoch, t, cutoff)
        previous = [i for i in self.rows if i[1] < t]
        prior = max(previous, key=lambda i: i[1]) if previous else None
        hold = deepcopy(self.rows[prior]) if prior else LiveExHold(media_fresh=True, fresh_sec=self.fresh_sec)
        hold.cached = None
        start = prior[1] if prior else -float('inf')
        applied = [k for k in self.keys if start < k['input_t'] <= t]
        for key in applied:
            hold.observe(key)
        self.applied = [r for k in applied for r in k.get('resets', [])]
        value, status = hold.available(now, t, cutoff, self.key_at(t), source, reason, self.epoch)
        self.rows[identity] = hold
        self._limit_rows()
        if value is not None:
            # 同一行の再選択でも、同じ列なら推論をやり直さない。
            mask = value['serial'][1]
            origins = tuple(value['hold_audit'].get('ex_hold_source_row_key_' + s)
                            for s in ('p1', 'p2'))
            value['serial'] = (identity, None if source is None else source['serial'], origins, mask)
        return value, status
