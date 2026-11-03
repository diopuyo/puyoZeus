"""24f の表示だけを切り替える。認識と各勝率の計算には値を戻さない。"""
from __future__ import annotations

import math
import os
from typing import Any

from .contracts import OverlaySnapshot


def enabled() -> bool:
    return os.environ.get('PUYO_LIVE_WINRATE_SWITCH') == '1' and os.environ.get('PUYO_LIVE_24H') != '1'


def probability(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1:
        return float(value)
    return None


def select_line(evaluations: dict, active: bool) -> dict:
    """鮮度判定済みの既存表示値を使い、独自の保持時間を足さない。"""
    practical = evaluations['practical']
    b3 = probability(practical.get('p1_win_probability'))
    v49f = probability((evaluations.get('v49f') or {}).get('p1'))
    ama = probability((evaluations.get('ama_plain') or {}).get('p1'))
    use_v49f = active and v49f is not None and practical.get('source') != 'confirmed_death'
    return dict(line1=v49f if use_v49f else b3,
                source1='v4.9f' if use_v49f else 'B3', b3=b3, v49f=v49f, ama=ama,
                v49f_active=active)


def with_switch(snapshot: OverlaySnapshot, row: dict) -> OverlaySnapshot:
    if not enabled():
        return snapshot
    payload = snapshot.to_mapping()
    payload['evaluations']['winrate_switch'] = select_line(
        payload['evaluations'], bool(row.get('_v49f_active')))
    terminal = payload.get('terminal', {})
    if terminal.get('state') == 'confirmed' and terminal.get('winner') in ('1P', '2P'):
        payload['evaluations']['winrate_switch'].update(
            line1=float(terminal['winner'] == '1P'), source1='confirmed_death')
    return OverlaySnapshot.from_mapping(payload)
