"""学習のHoldPolicyへ同じ解除事象を渡す配信用の薄い入出力変換。"""
from __future__ import annotations

import math
import os
from typing import Any

from .live_ex_fresh import EX_FRESH_SEC_DEFAULT, validate_ex_fresh_sec
from .live_ex_hold_policy import HoldPolicy, SIDES
from .live_ex_hold_diagnostics import record_rejection
from ._live_ex_frozen.src.dl1k_columns import COLUMNS, LEARNING_COLUMNS, blank

from ._live_ex_frozen.src.dl1k_retr import COLUMNS as RETR_COLUMNS, SECOND_COLUMNS
from ._live_ex_frozen.src.dl1k_xc import COLUMNS as XC_COLUMNS

HOLD_SEC = .5
FRESH_SEC = EX_FRESH_SEC_DEFAULT


class LiveExHold:
    def __init__(self, media_fresh: bool = False, fresh_sec: float = FRESH_SEC) -> None:
        self.fresh_sec = validate_ex_fresh_sec(fresh_sec)
        self.media_fresh = media_fresh
        self.policy = HoldPolicy(HOLD_SEC, fresh_sec=self.fresh_sec)
        self.columns, self.learning = COLUMNS, LEARNING_COLUMNS
        self.resets = {side: False for side in SIDES}
        self.cached: tuple | None = None
        self.value: dict | None = None
        self.expires: dict[str, float] = {}
        self.generation = 0
        self.pending_resets: list[dict] = []
        self.applied_resets: list[dict] = []

    def observe(self, key: dict) -> None:
        for child in getattr(self, '_light_holds', {}).values():
            if child is not None:
                child.observe(key)
        if os.environ.get('PUYO_LIVE_EX_AUDIT') == '1':
            self.pending_resets.extend(key.get('resets', []))
        for event in key.get('resets', []):
            side = event['side']
            self.resets[side] = True
            self.policy.clear(side)
            self.expires[side] = -math.inf
        if not key['fight']:
            for side in SIDES:
                self.policy.clear(side)
                self.expires[side] = -math.inf

    def adopt(self, event: dict, source: dict | None, status: str,
              skip_invalid: bool = False) -> dict:
        """再結合と同じ行構造。キャッシュ再使用は学習同様に無効。"""
        if source is not None:
            extra_columns = tuple(c for c in (*RETR_COLUMNS, *SECOND_COLUMNS, *XC_COLUMNS)
                                  if c in source['extra'] or c in self.policy.extra_columns)
            self.policy.extra_columns = extra_columns
            self.columns, self.learning = COLUMNS + extra_columns, LEARNING_COLUMNS + extra_columns
        extra = dict(source['extra']) if source is not None else blank(3)
        row = dict(group=event['group'], t=event['t'], row_key=event['row_key'],
                   fight=event['fight'], reset=event['reset'], status=status,
                   input_t=None if source is None else source['input_t'], extra=extra)
        if skip_invalid:
            try:
                self.policy.validate_original(row)
            except ValueError as error:
                record_rejection(str(error))
                row = dict(row, status='not_yet', input_t=None)
        return self.policy.apply(row)

    def available(self, now: float, input_t: float, cutoff: float, key: dict,
                  source: dict | None, reason: str, epoch: Any) -> tuple[dict | None, str]:
        result = self._available(now, input_t, cutoff, key, source, reason, epoch)
        from src.light_consumption import enabled
        if enabled() and not getattr(self, '_light_child', False):
            from .live_light_hold import available
            return available(self, result, now, input_t, cutoff, key, source, reason, epoch)
        return result

    def _available(self, now: float, input_t: float, cutoff: float, key: dict,
                   source: dict | None, reason: str, epoch: Any) -> tuple[dict | None, str]:
        identity = (epoch, input_t, cutoff)
        if identity != self.cached:
            self._attach(input_t, cutoff, key, source, reason, epoch)
            self.cached = identity
        if self.value is None:
            return None, reason
        extra = dict(self.value['extra'])
        for side in SIDES:
            if not self._usable(side, now, input_t):
                for name in self.learning:
                    if name.endswith('_' + side):
                        extra[name] = math.nan
        mask = tuple(math.isfinite(extra[name]) for name in self.learning)
        if not any(mask):
            return None, 'hold_missing_or_expired'
        value = dict(self.value, extra=extra, serial=(self.generation, mask))
        return value, 'held' if self.value['held'] else 'available'

    def _usable(self, side: str, now: float, t: float) -> bool:
        """同じ行の再評価でも媒体時刻の上下限と解除を守る。"""
        if not self.media_fresh:
            return not (now > self.expires.get(side, -math.inf))
        snap = self.policy.previous.get(side)
        if snap is None:
            return False
        mode = self.value['hold_audit'].get('ex_hold_mode_' + side)
        if mode == 'original':
            return 0 <= t - snap.input_t <= self.fresh_sec
        return mode in ('held', 'cache') and self.policy.usable(snap, t)

    def _attach(self, t: float, cutoff: float, key: dict, source: dict | None,
                reason: str, epoch: Any) -> None:
        game = key['key'][1] if key else None
        fight = key.get('fight', False) if key else False
        reset, self.resets = self.resets, {side: False for side in SIDES}
        self.applied_resets, self.pending_resets = self.pending_resets, []
        status = 'ok' if source is not None else reason
        status = {'not_ready_or_stale': 'not_yet', 'available': 'ok'}.get(status, status)
        event = dict(group=(epoch, game), t=t, row_key=str((epoch, t, cutoff)),
                     fight=fight, reset=reset)
        if source is not None:
            source = dict(source, extra=dict(source['extra']))
            for side in SIDES:
                # 選択済み行の非負の経過時間を引き、丸めで公開時刻が未来になるのを防ぐ。
                source['extra']['av_ex_' + side] = t - (cutoff - source['eligible_at'])
        values = self.adopt(event, source, status, skip_invalid=True)
        held = False
        for side in SIDES:
            mode = values['ex_hold_mode_' + side]
            snap = self.policy.previous.get(side)
            held |= mode in ('held', 'cache')
            if snap is not None:
                deadline = min(snap.attached_t + HOLD_SEC, snap.input_t + self.fresh_sec + HOLD_SEC)
                if mode == 'original':
                    deadline = snap.input_t + self.fresh_sec
                self.expires[side] = cutoff + deadline - t
        self.generation += 1
        self.value = dict(epoch=epoch, extra={c: values.get(c, math.nan) for c in self.columns}, held=held,
                          hold_audit={c: v for c, v in values.items() if c.startswith('ex_hold_')})
