"""撃ち返しの発火条件を固定する wrapper と、状態を持たない確率変換。"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

ENV_ENABLED = 'PUYO_COUNTER_FIRE_SHIFT'
TRANSFORM_PATH = Path(__file__).with_name('counter_fire_shift_1b.json')
TRANSFORM_SHA256 = 'd54c741adf59d6c8e8b86c36fa1328f3935a594e17c05984acc3e292b412e692'
WINDOW_SEC = 1.5
TIME_EPSILON = 1e-8
PROB_CLIP = 1e-6
HEIGHT_LIMIT = 12
MK_THRESHOLD = .5
ELIGIBLE_CELLS = (0, 1, 3, 4)
SMALLER_CELL_OFFSET = 3
NO_CELL = -1


def load_shift(path: Path = TRANSFORM_PATH, enabled: bool | None = None) -> dict | None:
    """採用設定を既定とし、ON のときだけ固定係数を検査する。"""
    if enabled is None:
        from src.production_config import DISTRIBUTION_COUNTER_FIRE_SHIFT
        enabled = os.environ.get(ENV_ENABLED, str(int(DISTRIBUTION_COUNTER_FIRE_SHIFT))) == '1'
    if not enabled:
        return None
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TRANSFORM_SHA256:
        raise ValueError(f'撃ち返し補正 JSON の SHA256 が一致しません: {path}')
    return json.loads(raw)


def number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    return math.nan if value is None else float(value)


def onset_cell(row: Mapping[str, Any], side: int) -> int:
    """PC2 と同じ順序で発火時の予測区分と高さを選ぶ。"""
    mk = number(row, f'mk_xc_p{side}')
    height = number(row, f'max_height_p{side}')
    if math.isnan(mk) or not math.isfinite(height):
        return NO_CELL
    top = int(height >= HEIGHT_LIMIT)
    if mk < MK_THRESHOLD:
        return top
    own = number(row, f'chain_pred_ojama_p{side}')
    other = number(row, f'chain_pred_ojama_p{3-side}')
    if math.isfinite(own) and math.isfinite(other) and own < other:
        return SMALLER_CELL_OFFSET + top
    return NO_CELL


def apply_shift(probability: float, sign: int, cell: int, model: Mapping | None) -> float:
    """温度較正後に一度だけ適用する。対象外は clip も行わない。"""
    if model is None or sign == 0 or cell not in ELIGIBLE_CELLS:
        return probability
    p = min(1-PROB_CLIP, max(PROB_CLIP, probability))
    x = math.log(p/(1-p)) + sign * model['values'][cell]
    return 1/(1+math.exp(-x))


class CounterFireShift:
    """時系列の予測行を観測し、同時刻の再推論では発火条件を保持する。"""

    def __init__(self, model: Mapping | None) -> None:
        self.model = model
        self.reset()

    def reset(self) -> None:
        self.previous: tuple[bool, bool] | None = None
        self.time = -math.inf
        self.events: dict[int, tuple[float, int]] = {}
        self.context = (0, NO_CELL)

    def observe(self, t: float, row: Mapping[str, Any]) -> tuple[int, int]:
        if self.model is None:
            return (0, NO_CELL)
        if t == self.time:
            return self.context
        if t < self.time:
            raise ValueError('撃ち返し補正の観測時刻が逆順です')
        active = tuple(number(row, f'chain_active_p{s}') > 0 for s in (1, 2))
        for side in (1, 2):
            if self.previous is None or not active[side-1] or self.previous[side-1]:
                continue
            self.events.pop(side, None)
            if active[2-side]:
                self.events[side] = (t, onset_cell(row, side))
        self.previous = active
        self.time = t
        candidates = [(start, side, cell) for side, (start, cell) in self.events.items()
                      if start <= t <= start+WINDOW_SEC+TIME_EPSILON]
        self.context = (0, NO_CELL)
        if candidates:
            latest = max(start for start, _, _ in candidates)
            owners = [(side, cell) for start, side, cell in candidates if start == latest]
            if len(owners) == 1:
                side, cell = owners[0]
                self.context = (1 if side == 1 else -1, cell)
        return self.context

    def apply(self, probability: float, t: float, row: Mapping[str, Any]) -> float:
        sign, cell = self.observe(t, row)
        return apply_shift(probability, sign, cell, self.model)
