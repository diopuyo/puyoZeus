"""D以前の結果を待機し、鮮度とキーを満たす最新の完成要求を選ぶ。"""
from __future__ import annotations

from collections import deque
from time import perf_counter

from .live_ex_async import AsyncEx, MAX_AGE_SEC
from .live_ex_fresh import validate_ex_fresh_sec
from ._live_ex_frozen.src.dl1k_columns import AVAILABLE_COLUMNS
from .live_ex_hold import LiveExHold
from .live_ex_row_time import RowTimeEx

MS = 1000.0


class DelayedAsyncEx(AsyncEx):
    def __init__(self, settings: dict) -> None:
        self.fresh_sec = validate_ex_fresh_sec(settings.get('ex_fresh_sec', MAX_AGE_SEC))
        self.delay_sec = settings['logical_delay_ms'] / MS
        self.completed: deque[dict] = deque()
        self.media_fresh = settings.get('media_fresh', False)
        self.row_time = RowTimeEx(fresh_sec=self.fresh_sec) if settings.get('row_time', False) else None
        self.xc_history = (self.row_time or RowTimeEx()) if settings.get('xc_same_time', False) else None
        self.latest_input_t = -float('inf')
        self.hold = LiveExHold(media_fresh=self.media_fresh, fresh_sec=self.fresh_sec)
        super().__init__(settings)

    def submit(self, packet: dict, epoch: int, captured_at: float) -> None:
        self.serial += 1
        self.epoch = epoch
        if getattr(self, 'xc_history', None) is not None:
            self.xc_history.boundary(epoch)
        if getattr(self, 'row_time', None) is not None:
            self.row_time.boundary(epoch)
        self.latest_input_t = packet['t']
        self.inputs.put(dict(packet=packet, epoch=epoch, serial=self.serial,
                             captured_at=captured_at, input_t=packet['t'], requested_at=perf_counter()))

    def poll(self) -> list[dict]:
        events = super().poll()
        now = perf_counter()
        for event in events:
            if event['kind'] == 'key' and hasattr(self, 'hold'):
                history = getattr(self, 'xc_history', None)
                if history is not None and history is not self.row_time:
                    history.observe(event)
                    history.prune(deque())
                if getattr(self, 'row_time', None) is not None:
                    self.row_time.observe(event)
                else:
                    self.hold.observe(event)
            if event['kind'] != 'result' or 'error' in event['value']:
                continue
            row = event['value']
            row['received_at'] = row['available_at']
            row['eligible_at'] = max(row['requested_at'] + self.delay_sec, row['received_at'])
            for name in AVAILABLE_COLUMNS:
                row['extra'][name] = row['eligible_at']
            self.completed.append(row)
        # 採用期間を延ばさない。境界を越えた完成物もここで捨てる。
        self._prune(now)
        return events

    def _prune(self, now: float) -> None:
        """旗ONでは最新入力の媒体時刻で完成物を掃除する。"""
        if getattr(self, 'row_time', None) is not None:
            self.completed = self.row_time.prune(self.completed)
            return
        media = getattr(self, 'media_fresh', False)
        before = len(self.completed)
        self.completed = deque(row for row in self.completed
                               if row['epoch'] == self.epoch and
                               (self.latest_input_t - row['input_t'] <= self.fresh_sec if media
                                else now - row['captured_at'] <= self.fresh_sec))
        self.pruned_count = getattr(self, 'pruned_count', 0) + before - len(self.completed)

    def available(self, now: float, input_t: float,
                  admitted_at: float | None = None) -> tuple[dict | None, str]:
        self.candidate_count = 0
        if getattr(self, 'media_fresh', False):
            self.latest_input_t = max(self.latest_input_t, input_t)
        self.poll()
        if getattr(self, 'row_time', None) is not None:
            cutoff = now if admitted_at is None else min(now, admitted_at)
            return self.row_time.select(self.completed, now, input_t, cutoff, self.error)
        if self.error:
            return None, 'worker_failed'
        if not self.key or self.key['serial'] != self.serial:
            return None, 'not_completed'
        if not self.key['fight']:
            return None, 'outside_exchange'
        cutoff = now if admitted_at is None else min(now, admitted_at)
        candidates = [row for row in self.completed if row['epoch'] == self.epoch
                      and row['key'] == self.key['key'] and row['fight']
                      and row['input_t'] <= input_t and row['eligible_at'] <= cutoff
                      and input_t - row['input_t'] <= self.fresh_sec
                      and (getattr(self, 'media_fresh', False)
                           or 0 <= now - row['captured_at'] <= self.fresh_sec)]
        if not candidates:
            return None, 'not_ready_or_stale'
        self.candidate_count = len(candidates)
        return max(candidates, key=lambda row: row['serial']), 'available'

    def held_available(self, now: float, input_t: float,
                       admitted_at: float | None = None) -> tuple[dict | None, str]:
        source, reason = self.available(now, input_t, admitted_at)
        self.selected_source = source
        cutoff = now if admitted_at is None else min(now, admitted_at)
        latest = getattr(self, 'latest', None) or {}
        same = bool(self.key and latest.get('key') == self.key['key'])
        self.availability_audit = dict(raw_reason=reason, latest_same_key=same,
            latest_wall_age_ms=(now-latest['captured_at'])*MS if same else None,
            latest_media_age_ms=(input_t-latest['input_t'])*MS if same else None)
        if getattr(self, 'row_time', None) is not None:
            return self.row_time.held(now, input_t, cutoff, source, reason)
        return self.hold.available(now, input_t, cutoff, self.key or {},
                                   source, reason, self.epoch)

    def protect_rows(self, oldest_t: float, epoch: int) -> None:
        """再評価中の行と、B3がまだ受理し得る行を掃除から保護する。"""
        if getattr(self, 'row_time', None) is not None:
            self.row_time.boundary(epoch)
            self.row_time.floor = oldest_t
        if getattr(self, 'xc_history', None) is not None:
            self.xc_history.boundary(epoch)
            self.xc_history.floor = oldest_t

    def selection_record(self, t: float, value: dict | None) -> dict:
        """環境変数で要求されたときだけ呼ぶ監査用の状態。"""
        history = getattr(self, 'row_time', None)
        key = history.key_at(t) if history is not None else (self.key or {})
        source = getattr(self, 'selected_source', None) or {}
        pruned = history.pruned if history is not None else getattr(self, 'pruned_count', 0)
        result = dict(key_serial=key.get('serial'), key_fight=key.get('fight'),
                      latest_input_t=self.latest_input_t,
                      candidates=history.candidates if history else self.candidate_count,
                      result_serial=source.get('serial'), result_input_t=source.get('input_t'),
                      hold={} if value is None else value.get('hold_audit', {}),
                      resets=history.applied if history else (self.hold.applied_resets + self.hold.pending_resets),
                      pruned=pruned, row_time=history is not None)
        if history is not None:
            result.update(overflow_keys=history.overflow_keys,
                          overflow_results=history.overflow_results,
                          overflow_rows=history.overflow_rows)
            history.pruned = 0
        self.pruned_count = 0
        return result
