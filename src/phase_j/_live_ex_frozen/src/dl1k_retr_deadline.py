"""連鎖後の最初の設置で再発火できる期限。既定OFF。"""
from __future__ import annotations

import math
import os

ENV = 'PUYO_RETR_REFIRE_DEADLINE'
NEXT_PLACEMENT = 1
EPSILON = 1e-9


def enabled() -> bool:
    return os.environ.get(ENV, '0') == '1'


def hands(end: float, attacker_end: float, seconds_per_hand: float, maximum: int) -> int:
    """予告が確定していても、連鎖後の最初の置きで発火すれば降らない。"""
    before = math.floor((attacker_end-end+EPSILON)/seconds_per_hand)
    return min(maximum, max(NEXT_PLACEMENT, before+NEXT_PLACEMENT))
