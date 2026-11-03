"""再結合用の強い入力鍵と、全通知から原向き行へ集約する解除事象。"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from types import BuiltinFunctionType
from typing import Any

import numpy as np

SCHEMA = 1
SIDES = ('p1', 'p2')
EVENTS = ('fire', 'landing', 'ojama_fall', 'boundary', 'target')
# 実体は資産SHAに含める。キャッシュの内容や呼出し回数は意味入力ではない。
EXCLUDED = frozenset(('native', 'lookup', 'lean', 'native_cache', 'counters'))
INPUT_FIELDS = ('state', 'chains', 'histories', 'ledger_history', 'tracker', 'knowledge', 'placements')


def canonical(value: Any) -> Any:
    """未知の型は黙って文字列化せず停止する。順序と浮動小数点を保存する。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return ['float', float(value).hex()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return ['array', str(value.dtype), list(value.shape), value.tobytes().hex()]
    if isinstance(value, bytes):
        return ['bytes', value.hex()]
    if isinstance(value, BuiltinFunctionType) and value.__module__.split('.')[0] == 'puyo_core':
        return ['native_function', value.__module__, value.__qualname__]
    if isinstance(value, dict):
        pairs = [[canonical(k), canonical(v)] for k, v in value.items()]
        return ['dict', sorted(pairs, key=lambda p: json.dumps(p[0]))]
    if isinstance(value, (list, tuple, deque)):
        return [canonical(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return ['set', sorted((canonical(v) for v in value), key=json.dumps)]
    if hasattr(value, '__dict__'):
        fields = {k: v for k, v in vars(value).items() if k not in EXCLUDED}
        return [type(value).__module__+'.'+type(value).__qualname__, canonical(fields)]
    raise TypeError(f'強い鍵に未対応の型: {type(value)!r}')


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(canonical(value), separators=(',', ':'),
                                    ensure_ascii=False).encode('utf-8')).hexdigest()


def input_keys(engine: Any, assets: dict, generations: list[dict]) -> dict[str, str]:
    """両側に依存する計算を物理側・解除世代とともに識別する。論理時刻も省かない。"""
    state = {name: getattr(engine, name) for name in INPUT_FIELDS}
    state['proof_state'] = proof_state(engine)
    base = digest(dict(schema=SCHEMA, assets=assets, epoch=engine.epoch,
                       policy=engine.policy, inputs=state))
    return {side: digest((base, side, generations[i])) for i, side in enumerate(SIDES)}


def proof_state(engine: Any) -> dict:
    """探索の前後で変わり、後続要求にも効く死亡証明の状態。派生キャッシュは除く。"""
    return deepcopy(dict(verdicts=engine.tracker.verdicts, physics=engine.tracker.physics,
                         certs=engine.landing.certs, game=engine.landing.game))


def restore_proof_state(engine: Any, value: dict) -> None:
    saved = deepcopy(value)
    engine.tracker.verdicts, engine.tracker.physics = saved['verdicts'], saved['physics']
    engine.landing.certs, engine.landing.game = saved['certs'], saved['game']


def finite(value: Any) -> Any:
    return None if isinstance(value, float) and not math.isfinite(value) else value


class EventMonitor:
    """行の間で始まって終わる事象も世代差として保持する。"""

    def __init__(self) -> None:
        self.previous: tuple | None = None
        self.generations = [{name: 0 for name in EVENTS} for _ in SIDES]
        self.pending: list[dict] = []

    def observe(self, engine: Any, frame: Any) -> None:
        probe = engine.probe()
        sides = engine.state.sides
        target = (probe.fight, tuple(engine._resolving(i) for i in range(len(SIDES))))
        current = (probe.key[:2], target, probe.key[2],
                   tuple((s.land_count, finite(s.landing.finalize_t) if s.landing else None) for s in sides))
        old = self.previous
        for i, side in enumerate(SIDES):
            flags = dict(boundary=old is None or current[0] != old[0],
                         target=old is None or current[1:3] != old[1:3],
                         fire=current[2][i] is not None and (old is None or current[2][i] != old[2][i]),
                         landing=old is not None and current[3][i] != old[3][i],
                         ojama_fall=sides[i].state == 'OJAMA_FALL')
            for name, yes in flags.items():
                if yes:
                    self.generations[i][name] += 1
                    event_t = current[2][i] if name == 'fire' else frame.t
                    self.pending.append(dict(side=side, event=name, t_sec=frame.t,
                                             event_t_sec=event_t, index=frame.index))
        self.previous = current

    def take(self) -> list[dict]:
        result, self.pending = self.pending, []
        return result
