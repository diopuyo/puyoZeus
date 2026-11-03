"""EXの元入力からの鮮度。保持時間とは独立に設定する。"""
from __future__ import annotations

import math

EX_FRESH_SEC_DEFAULT = 0.5


def validate_ex_fresh_sec(value: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('live_24h_ex_fresh_secは0より大きい有限な秒数です')
    return float(value)
