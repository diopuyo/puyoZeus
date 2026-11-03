"""通知時点の retR を、媒体時刻を固定して死亡判定へ渡す。"""
from __future__ import annotations

from contextvars import ContextVar
import math
import os
from typing import Any
from .live_ex_hold_policy import EX_HOLD_MAX_SEC
from .live_ex_hold import LiveExHold

ENV_ENABLED = 'PUYO_LIVE_DEATH_CHECK_RETR'
CURRENT: ContextVar[dict | None] = ContextVar('live_death_retr', default=None)
UPDATE_ARGUMENTS = 8
UPDATE_DEFAULTS = ((None, None), None, (False, False))


class DeathRetrReader:
    """表示モデルの hold を変更せず、死亡判定専用に同じ規則で保持する。"""

    def __init__(self, fresh_sec: float) -> None:
        self.fresh_sec = fresh_sec
        self.hold = LiveExHold(media_fresh=True, fresh_sec=fresh_sec)
        self.epoch: int | None = None
        self.last_t = -math.inf

    def select(self, worker: Any, now: float, t: float, cutoff: float) -> dict | None:
        source, reason = worker.available(now, t, admitted_at=cutoff)
        if self.epoch != worker.epoch or t < self.last_t:
            self.hold = LiveExHold(media_fresh=True, fresh_sec=self.fresh_sec)
            self.epoch, self.last_t = worker.epoch, -math.inf
        history = getattr(worker, 'row_time', None)
        key = history.key_at(t) if history is not None else (worker.key or {})
        keys = history.keys if history is not None else ([key] if key else [])
        for observed in keys:
            if self.last_t < observed['input_t'] <= t:
                self.hold.observe(observed)
        self.last_t = t
        value, _ = self.hold.available(now, t, min(now, cutoff), key, source, reason, worker.epoch)
        return value


def capture(b3: Any, notice: Any) -> None:
    """EX 自身の hold 選択を使い、将来の結果を持ち込まない。"""
    CURRENT.set(None)
    unified = getattr(b3, 'live24h', None)
    if os.environ.get(ENV_ENABLED) != '1' or unified is None or unified.worker is None:
        return
    frame, _, _, epoch = notice.b3
    changed = epoch != b3.epoch
    adopted = None if changed else b3.adopted_game
    generation = b3.generation + int(changed)
    generation += int(frame.get('adopted_game') is not None and frame['adopted_game'] != adopted)
    if unified.epoch != generation or unified.game != frame['game_idx']:
        return
    worker = unified.worker
    if not hasattr(unified, 'death_retr_reader'):
        unified.death_retr_reader = DeathRetrReader(worker.fresh_sec)
    value = unified.death_retr_reader.select(worker, unified.clock(), frame['t_sec'], notice.captured_at)
    CURRENT.set(dict(t_sec=frame['t_sec'], game=frame['game_idx'],
                     values=tuple(side_values(value, s, frame['t_sec'], worker.fresh_sec)
                                  for s in ('p1', 'p2'))))


def side_values(value: dict | None, side: str, t: float, fresh_sec: float) -> tuple:
    """壁時計の hold 設定でも、元入力の媒体時刻の上下限を確認する。"""
    if value is None:
        return math.nan, math.nan
    audit = value.get('hold_audit', {})
    source_t = audit.get('ex_hold_source_input_t_' + side)
    mode = audit.get('ex_hold_mode_' + side)
    limit = fresh_sec + (EX_HOLD_MAX_SEC if mode in ('held', 'cache') else 0)
    if not isinstance(source_t, (int, float)) or not 0 <= t - source_t <= limit:
        return math.nan, math.nan
    extra = value['extra']
    return extra.get('retR_r_' + side, math.nan), extra.get('retR_c_' + side, math.nan)


def with_snapshot(args: tuple) -> tuple:
    """IPC の journal にも当時の入力を残す。OFF では引数も変更しない。"""
    if os.environ.get(ENV_ENABLED) != '1' or len(args) > UPDATE_ARGUMENTS:
        return args
    defaults = UPDATE_DEFAULTS[len(args) - 5:]
    return (*args, *defaults, CURRENT.get())


def audit_value(value: Any) -> Any:
    """判定用の値を変えず、監査記録の非有限値を JSON の null にする。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: audit_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(audit_value(item) for item in value)
    return value


def can_return(snapshot: dict | None, side: int, required: float, t_sec: float) -> bool:
    """選択済みの同時刻の有限値だけを死亡の反証とする。"""
    if snapshot is None or snapshot['t_sec'] != t_sec:
        return False
    residual, cancel = snapshot['values'][side]
    return ((isinstance(residual, (int, float)) and math.isfinite(residual) and residual <= 0)
            or (isinstance(cancel, (int, float)) and math.isfinite(cancel) and cancel >= required))
