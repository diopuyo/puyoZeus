"""到着した発火入力・式・得点・盤面だけで終了盤候補を逐次更新する。"""
from __future__ import annotations

from time import perf_counter
from typing import Any

MS = 1000.
RECORD_NAMES = ('before', 'after')


class OriginStream:
    def __init__(self, policy: dict, native: Any, record_budget: bool = False) -> None:
        self.policy, self.native = policy, native
        self.active: dict[int, tuple] = {}
        self.game: int | None = None
        # 段1 (既定OFF): 予測時間の例外門 (MAX_LOW_SPEC_MS) を記録だけにする。
        self.record_budget = record_budget
        self.budget_exceeded: list[float] = []

    def observe(self, game: int, stamp: float, messages: list[dict]) -> list[dict]:
        """同時刻の順序も保存原票と同じにする。入力時刻より先の配送を拒否する。"""
        if game != self.game:
            self.game, self.active = game, {}
        values = []
        for message in messages:
            if message['t'] > stamp:
                raise ValueError('未到着の発火入力')
            value = self.dispatch(message)
            if value is not None:
                values.append((message['order'], value))
        return [v for _, v in sorted(values, key=lambda pair: (pair[1]['input_sec'], pair[0]))]

    def dispatch(self, message: dict) -> dict | None:
        if not self.record_budget:
            return self.dispatch_checked(message)
        return self.dispatch_recorded(message)

    def dispatch_checked(self, message: dict) -> dict | None:
        from src.prefire_origin_stream import initialize, on_formula, on_score, terminal_observation, snapshot
        from src.prefire_worker import LOW_SPEC_RATIO
        began = perf_counter()
        key, kind, data = message['id'], message['kind'], message['data']
        if kind == 'initial':
            state = initialize(data, self.policy, self.native)
            self.active[key] = (data, state, message.get('initial_next'))
            elapsed = ((perf_counter()-began)*MS+
                       sum(s.get('image_compute_ms', 0.) for s in data['sources']))*LOW_SPEC_RATIO
            return snapshot(state, data, self.policy, elapsed, True)
        inp, state, initial_next = self.active[key]
        length = len(state['after'])
        if kind == 'formula':
            on_formula(state, data, inp, self.policy, self.native)
        elif kind == 'score':
            on_score(state, data, inp, self.policy)
        elif kind == 'frame':
            return terminal_observation(state, data, initial_next, inp, self.policy, self.native)
        else:
            raise ValueError(f'未知の撃ち合い通知: {kind}')
        if len(state['after']) > length:
            return snapshot(state, inp, self.policy, (perf_counter()-began)*MS*LOW_SPEC_RATIO, False)
        return None

    def dispatch_recorded(self, message: dict) -> dict | None:
        """門の上限だけを呼出しの間だけ外し (終了時に戻す)、超過は原票へ残す。計算は同じ。"""
        from src import prefire_origin_stream as stream
        previous = None if message['kind'] == 'initial' else self.active.get(message['id'])
        lengths = {name: len(previous[1][name]) for name in RECORD_NAMES} if previous else {}
        limit, stream.MAX_LOW_SPEC_MS = stream.MAX_LOW_SPEC_MS, float('inf')
        try:
            value = self.dispatch_checked(message)
        finally:
            stream.MAX_LOW_SPEC_MS = limit
        current = self.active.get(message['id'])
        if current is None:
            return value
        state = current[1]
        lows = [r['low_spec_ms'] for name in RECORD_NAMES for r in state[name][lengths.get(name, 0):]]
        if message['kind'] == 'initial':
            lows.append(state['initial']['low_spec_ms'])
        over = [low for low in lows if low > limit]
        if not over:
            return value
        self.budget_exceeded.append(max(over))
        return value if value is None else dict(value, budget_exceeded_ms=max(over))

    def take_budget_exceeded(self) -> list[float]:
        """前回の取得後に記録した超過 (低スペック換算ms) を渡して空にする。"""
        values, self.budget_exceeded = self.budget_exceeded, []
        return values
