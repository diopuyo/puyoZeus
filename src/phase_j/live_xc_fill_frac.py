"""選択・保持後の xc から、学習と同じ精度で致死量の割合を作る。"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np

XC_FILL_FRAC_COLUMNS = ('xc_fill_frac_p1', 'xc_fill_frac_p2')


def fill_fractions(values: Mapping[str, float], columns: Iterable[str]) -> dict[str, float]:
    """モデルが使う列だけ計算する。左右は既存 xc の向きをそのまま使う。"""
    result = {}
    for name in columns:
        side = name.rsplit('_', 1)[1]
        net = float(values.get(f'xc_net_recv_{side}', math.nan))
        room = float(values.get(f'xc_death_room_{side}', math.nan))
        denominator = net + room
        if math.isnan(net) or math.isnan(room):
            value = math.nan
        else:
            value = net / denominator if denominator > 0 else 1.0
        # JSON にも渡せる Python float に戻すが、値は float32 の丸めを保つ。
        result[name] = float(np.float32(value))
    return result
