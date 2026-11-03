"""v4.1レビュー専用。凍結盤の鮮度を外部時刻から適用し、本番既定へ接続しない。"""
from __future__ import annotations
from math import exp, isfinite
import numpy as np
from src.prefire_display import logistic

V2_SLOPE = .0020023091162859226
V2_C = .5


def fresh_rows(rows: list[dict], decay: float) -> list[dict]:
    """盤面は凍結のまま保持し、時間経過に伴う信頼度だけを下げる。0は旧挙動。"""
    if not isfinite(decay) or decay < 0:
        raise ValueError('鮮度減衰率は有限の非負値が必要です')
    result = []
    for row in rows:
        stamp = row.get('input_sec')
        if stamp is not None and stamp > row['t']:
            raise ValueError('未来の確定盤')
        weight = exp(-decay*max(0., row['t']-stamp)) if stamp is not None else 0.
        result.append(dict(row, a=row['a']*weight, b=row['b']*weight))
    return result


def predict(rows: list[dict], model: dict) -> np.ndarray:
    fresh = fresh_rows(rows, model['decay'])
    values = [r['b3'] if r.get('terminal', False) else np.clip(r['b3']+model['c']*(
        logistic(model['attack_beta']*r['a']-model['return_beta']*r['b'])-.5), 0., 1.) for r in fresh]
    return np.asarray(values)


def implied_balance(rows: list[dict], model: dict) -> np.ndarray:
    """両目的を同じV2の個数軸へ戻す。cによる縮小前の潜在収支。"""
    fresh = fresh_rows(rows, model['decay'])
    return np.asarray([(model['attack_beta']*r['a']-model['return_beta']*r['b'])/V2_SLOPE for r in fresh])
