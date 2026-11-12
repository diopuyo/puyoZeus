"""中盤補正の時系列状態。盤の履歴は同じ試合の過去だけを参照する。"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Mapping

from src.midgame_board_layer import ZERO_FIELDS, board_shift, eligible, settled_features, shift_probability

SIDES = (1, 2)


@dataclass(frozen=True)
class MixingState:
    last: float = 0.0
    entered: float | None = None
    exited: float | None = None
    inside: bool = False


def mix_shift(target: float | None, t: float, state: MixingState, model: Mapping,
              counter_window: bool = False) -> tuple[float, MixingState]:
    """v2どおり、退出した行を減衰の起点にし、再進入時は0から上げる。"""
    if target is not None and not counter_window:
        entered = state.entered if state.inside else t
        value = target * min(1.0, (t - entered) / model['enter_sec'])
        return value, MixingState(value, entered, state.exited, True)
    exited = t if state.inside else state.exited
    last = 0.0 if counter_window else state.last
    value = last * math.exp(-(t - exited) / model['exit_sec']) if exited is not None else 0.0
    return value, MixingState(last, state.entered, exited, False)


class MidgameBoardLayer:
    def __init__(self, model: Mapping | None) -> None:
        self.model = model
        self.reset()

    def reset(self) -> None:
        self.times: list[list[float]] = [[], []]
        self.features: list[list[tuple[int, int]]] = [[], []]
        self.observed_time = -math.inf
        self.time = -math.inf
        self.shift = 0.0
        self.before = MixingState()
        self.mixing = MixingState()

    def observe(self, frame: Mapping[str, Any]) -> None:
        """行より未来の盤を使わないよう、変化した置き終わり特徴だけ保存する。"""
        if self.model is None:
            return
        t = float(frame['t_sec'])
        if t < self.observed_time:
            raise ValueError('中盤補正の盤の観測時刻が逆順です')
        self.observed_time = t
        for side in frame.get('sides', ()):
            index = {'1P': 0, '2P': 1}.get(side.get('side'))
            if index is None:
                continue
            feature = settled_features(side.get('confirmed_board'))
            if feature is None or (self.features[index] and self.features[index][-1] == feature):
                continue
            if self.times[index] and self.times[index][-1] == t:
                self.features[index][-1] = feature
            else:
                self.times[index].append(t)
                self.features[index].append(feature)

    def target(self, t: float, row: Mapping[str, Any], counter_window: bool) -> float | None:
        if not eligible(row, counter_window):
            return None
        indices = [bisect_right(times, t) - 1 for times in self.times]
        if min(indices) < 0:
            return None
        return board_shift(self.features[0][indices[0]], self.features[1][indices[1]], self.model)

    def apply(self, probability: float, t: float, row: Mapping[str, Any],
              counter_window: bool = False) -> float:
        """進入は線形、退出は指数減衰。counter_window (撃ち返しの窓・撃ち合いの開始・光の候補) では残りも即時ゼロ。"""
        if self.model is None:
            return probability
        if t < self.time:
            raise ValueError('中盤補正の推論時刻が逆順です')
        if t != self.time:
            self.before = self.mixing
            self.time = t
        target = self.target(t, row, counter_window)
        self.shift, self.mixing = mix_shift(target, t, self.before, self.model, counter_window)
        return shift_probability(probability, self.shift)


def exchange_started(row: Mapping[str, Any]) -> bool:
    """連鎖・予告・おじゃまのどれかが出たら撃ち合いの局面。欠測は撃ち合いと扱わない。"""
    for name in ZERO_FIELDS:
        for side in SIDES:
            try:
                if float(row.get(f'{name}_p{side}') or 0) != 0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def apply_midgame(owner: Any, probability: float, result: Mapping, values: Mapping) -> float:
    """Live24hの接続部を小さく保つ。撃ち返しの窓・撃ち合いの開始・光の候補では残りも即時ゼロ。"""
    layer = getattr(owner, 'midgame_board_layer', None)
    if layer is None or layer.model is None:
        return probability
    from src.counter_fire_shift import TIME_EPSILON, WINDOW_SEC
    counter = getattr(owner, 'counter_fire_shift', None)
    t = result['t_sec']
    row = dict(result['row'], **values)
    window = (counter is not None and counter.model is not None and
              any(start <= t <= start + WINDOW_SEC + TIME_EPSILON
                  for start, _ in counter.events.values()))
    hard_exit = window or bool(values.get('_light_scenarios')) or exchange_started(row)
    return layer.apply(probability, t, row, hard_exit)
