"""v4の任意後段表示。発火前後を単一項で接続し、旧世代を拒否する。"""
from __future__ import annotations
from dataclasses import dataclass, field
from math import exp
import numpy as np
from src.prefire_trial_defaults import PREFIRE_V4_ENABLED_DEFAULT

EPS = 1e-6
COUNT_UNIT = 30.


def logistic(value: float) -> float:
    return 1./(1.+exp(-max(-700., min(700., value))))


def correction(baseline: float, sides: list[dict], model: dict) -> float:
    """B3 offsetでfitした傾きを、指定の加算形c・g(x)へ適用する。"""
    a = (sides[0]['attack']-sides[1]['attack'])/COUNT_UNIT
    b = (sides[0]['returned']-sides[1]['returned'])/COUNT_UNIT
    return model['c']*(logistic(model['attack_beta']*a-model['return_beta']*b)-.5)


@dataclass
class DisplayState:
    """状態は外部ラッパーに限る。到着前・窓終了後の古い結果は使わない。"""
    enabled: bool = PREFIRE_V4_ENABLED_DEFAULT
    game: int | None = None
    generation: int = 0
    mode: str = 'prefire'
    target: float = 0.
    start_value: float = 0.
    changed_sec: float = 0.
    tau: float = 1.
    history: list[tuple[float, float]] = field(default_factory=list)

    def value(self, stamp: float) -> float:
        if not self.enabled:
            return 0.
        fraction = min(1., max(0., (stamp-self.changed_sec)/self.tau)) if self.tau else 1.
        return self.start_value+(self.target-self.start_value)*fraction

    def reset(self, game: int, stamp: float) -> None:
        self.game, self.generation, self.mode = game, 0, 'prefire'
        self.target, self.start_value, self.changed_sec = 0., 0., stamp
        self.history.clear()

    def switch(self, mode: str, stamp: float, generation: int) -> None:
        self.start_value = self.value(stamp)
        self.target, self.changed_sec = self.start_value, stamp
        self.mode, self.generation = mode, generation

    def accept(self, result: dict, delta: float, stamp: float, tau: float) -> bool:
        if not self.enabled or result['game'] != self.game or result['generation'] != self.generation:
            return False
        if result['mode'] != self.mode or result.get('over_budget') or result['ready_sec'] > stamp:
            return False
        if any(t > result['input_sec'] for t in result['evidence_times']):
            raise ValueError('未来参照')
        if result['ready_sec'] < result['input_sec']+result['low_spec_ms']/1000.:
            raise ValueError('計算時間の前倒し')
        self.start_value, self.target = self.value(stamp), delta
        self.changed_sec, self.tau = stamp, tau
        self.history.append((stamp, delta))
        return True

    def display(self, baseline: float, stamp: float) -> float:
        return float(np.clip(baseline+self.value(stamp), 0., 1.)) if self.enabled else baseline


@dataclass
class ModeBlend:
    """レビューとlive接続用の共通逐次接続器。過去の出力は更新しない。"""
    enabled: bool = PREFIRE_V4_ENABLED_DEFAULT
    game: object = None
    mode: bool = False
    previous: float = 0.
    start: float = 0.
    origin: float = 0.

    def step(self, game: object, stamp: float, mode: bool, target: float | None, tau: float) -> float:
        if not self.enabled:
            return 0.
        if tau <= 0:
            raise ValueError('接続時間は正が必要です')
        if self.game != game:
            self.game, self.mode, self.previous = game, mode, 0.
            self.start, self.origin = stamp-tau, 0.
        if self.mode != mode:
            self.mode, self.start, self.origin = mode, stamp, self.previous
        target = self.previous if target is None else target
        fraction = min(1., max(0., (stamp-self.start)/tau))
        self.previous = self.origin+(target-self.origin)*fraction
        return self.previous
