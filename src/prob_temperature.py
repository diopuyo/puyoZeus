"""配信の勝率を、PC2 の層別温度で較正する (画面調整とは独立)。"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from src import production_config

PROB_CLIP = 1e-6
EARLY_MAX = 18
LATE_MIN = 48
DIFF_LIMITS = (2, 5, 9)
DIFF_LARGE_MIN = 10
NO_BAND = -1
PHASE_NAMES = ('序盤', '中盤', '終盤')
EXCHANGE_NAMES = ('撃ち合いなし', '撃ち合い中')
DIFF_NAMES = ('0〜2', '3〜5', '6〜9', '10 以上')
TRANSFORM_SHA256 = '5bb39a21f8e3c7a50e104aba888835fd512005f0f78d9afeb76534c56a317585'
TRANSFORM_PATH = Path(__file__).resolve().parents[1] / 'models/calibration/prob_temperature_v1.json'
ENV_ENABLED = 'PUYO_DISTRIBUTION_PROB_TEMPERATURE'


def load_temperature(path: Path = TRANSFORM_PATH, enabled: bool | None = None) -> dict | None:
    """ON なら起動時に読み、欠落・改変をその場でエラーにする。"""
    if enabled is None:
        enabled = os.environ.get(ENV_ENABLED, str(int(
            production_config.DISTRIBUTION_PROB_TEMPERATURE))) == '1'
    if not enabled:
        return None
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TRANSFORM_SHA256:
        raise ValueError(f'勝率の温度 JSON の SHA256 が一致しません: {path}')
    return json.loads(raw)


def apply_temperature(probability: float, row: Mapping[str, Any], model: Mapping | None) -> float:
    """平均後の生の勝率に一度だけ適用する、状態を持たない変換。"""
    if model is None:
        return probability
    values = {name: math.nan if row.get(name) is None else float(row[name])
              for name in ('cells_total_p1', 'cells_total_p2', 'cells_color_p1',
                           'cells_color_p2', 'chain_active_p1', 'chain_active_p2')}
    temperature = model['T0']
    totals = (values['cells_total_p1'], values['cells_total_p2'])
    if all(math.isfinite(value) for value in totals):
        phase = max(0 if value <= EARLY_MAX else 2 if value >= LATE_MIN else 1
                    for value in totals)
        # 11/9 Claude 決定: 学習時の calib.py に合わせ、NaN > 0 は偽とする。
        exchange = int(values['chain_active_p1'] > 0 or values['chain_active_p2'] > 0)
        key = f'{PHASE_NAMES[phase]}・{EXCHANGE_NAMES[exchange]}'
        if model['kind'] >= 2:
            temperature = model['T2'][key]
        difference = abs(values['cells_color_p1'] - values['cells_color_p2'])
        # calib.py の np.select と同じく、非整数の 9 < 差 < 10 には層がない。
        band = next((i for i, limit in enumerate(DIFF_LIMITS) if difference <= limit),
                    len(DIFF_LIMITS) if difference >= DIFF_LARGE_MIN else NO_BAND)
        if model['kind'] >= 3 and math.isfinite(difference) and band != NO_BAND:
            temperature = model['T3'][f'{key}・差 {DIFF_NAMES[band]}']
    clipped = min(max(probability, PROB_CLIP), 1 - PROB_CLIP)
    logit = math.log(clipped / (1 - clipped))
    return 1 / (1 + math.exp(-temperature * logit))
