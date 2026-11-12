"""候補ごとに従来の保持規則を適用し、異なる候補のEXを混ぜない。"""
from __future__ import annotations
from typing import Any
import math


def identity(row: dict) -> tuple:
    return tuple(None if side is None else (side['chain_id'], side['total'],
                 tuple(side.get('steps', ())), tuple(tuple(r) for r in side.get('final', ())))
                 for side in row['sides'])


def available(owner: Any, base: tuple, now: float, input_t: float, cutoff: float,
              key: dict, source: dict | None, reason: str, epoch: Any) -> tuple:
    from .live_ex_hold import LiveExHold
    if source is not None:
        # 行時刻ごとの保持を複製するとき、使わない探索木まで複製しない。
        # 候補・重み・EX列は残し、元の計算結果には手を加えない。
        owner._light_choices = [dict(row, result={}) for row in source.get('light_scenarios', [])]
    choices = getattr(owner, '_light_choices', [])
    if not choices:
        owner._light_holds = {}
        return base
    holds = getattr(owner, '_light_holds', {})
    owner._light_holds = {identity(row): holds.get(identity(row)) for row in choices}
    results = []
    for row in choices:
        token = identity(row)
        child = owner._light_holds[token]
        if child is None:
            child = LiveExHold(owner.media_fresh, owner.fresh_sec)
            child._light_child = True
            owner._light_holds[token] = child
        selected = None if source is None else dict(source, extra=row['extra'], result=row['result'])
        value, why = child.available(now, input_t, cutoff, key, selected, reason, epoch)
        extra = value['extra'] if value is not None else dict.fromkeys(owner.learning, math.nan)
        results.append(dict(row, extra=extra, hold_audit={} if value is None else value.get('hold_audit', {})))
    value, why = base
    if value is not None:
        value = dict(value, light_scenarios=results)
    return value, why
