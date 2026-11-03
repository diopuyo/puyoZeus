"""採用したB3行から公開用の生値だけを抜き出す。再計算・正規化はしない。"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.b3_column_spec import COLUMN_SPECS, Mask

SIDE_FIELDS = ('sl_pending', 'sl_generated', 'sl_offset', 'chain_active',
               'chain_n_steps', 'chain_pred_l', 'chain_rem_stages', 'chain_rem_sec',
               'k1_ojama', 'cells_color', 'cells_ojama', 'mk_sl', 'mk_chain', 'mk_k')
TIME_FIELDS = ('ta_H_med', 'ta_q_med', 'ta_m_sl_med', 'ta_bound', 'mk_ta')
FLOAT32_COLUMNS = frozenset(c.name for c in COLUMN_SPECS if c.dtype == 'f4')


def finite(value: Any) -> float | None:
    """JSONにNaNを出さず、欠測をnullに統一する。"""
    return float(value) if value is not None and math.isfinite(value) else None


def column_value(row: dict, name: str) -> float | None:
    """保存学習行と同じ列型へ丸め、公開値も完全一致させる。"""
    value = finite(row.get(name))
    return float(np.float32(value)) if value is not None and name in FLOAT32_COLUMNS else value


def contents_from_row(row: dict) -> dict:
    """時間軸だけ相手の攻撃列を参照し、全項目を受け側基準に揃える。"""
    result = dict(frame=int(row['frame_idx']), t_sec=float(row['t_sec']),
                  model='exchange' if row['mode_exchange'] else 'static',
                  sl_rate=finite(row.get('sl_rate')))
    for side, attacker in (('p1', 'p2'), ('p2', 'p1')):
        values = {key: column_value(row, f'{key}_{side}') for key in SIDE_FIELDS}
        values.update({key: column_value(row, f'{key}_{attacker}') for key in TIME_FIELDS})
        generated, offset = values['sl_generated'], values['sl_offset']
        values['sl_sent'] = None if generated is None or offset is None else generated-offset
        masks = [values[key] for key in ('mk_chain', 'mk_ta', 'mk_k')]
        values['unpredictable'] = (True if Mask.UNPREDICTABLE in masks else
                                   None if all(mask is None for mask in masks) else False)
        result[side] = values
    return result
