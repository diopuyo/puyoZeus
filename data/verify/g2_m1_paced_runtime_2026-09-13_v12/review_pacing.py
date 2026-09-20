"""成功runの休止保存票を全フレーム・元driver開始票へ結ぶ。熱安全性の認証ではない。"""
from __future__ import annotations
import json
import math
from pathlib import Path

FIRST, LAST, STRIDE = 29052, 36298, 2
MAX_SLEEP_SECONDS = 1.0


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('review_pacing:' + reason)


def number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def verify(output: Path) -> dict:
    value = json.loads((output / 'FRAME_PACING_STATUS.json').read_bytes())
    driver = json.loads((output / 'WHOLE_M1_DRIVER_STATUS.json').read_bytes())
    require(value['closed'] is True and value['error'] is None and value['body_error'] is None, 'closed')
    require(value['quality_gate_clear'] is False and value['max_sleep_seconds'] == MAX_SLEEP_SECONDS, 'authority')
    start = driver['start_frame']
    require(type(start) is int and FIRST <= start < LAST and (start - FIRST) % STRIDE == 0, 'start')
    expected, rows = tuple(range(FIRST, LAST, STRIDE)), value['rows']
    require(tuple(row['frame'] for row in rows) == expected, 'frames')
    for row in rows:
        work, requested, actual = (row[key] for key in ('work_seconds', 'requested_sleep_seconds', 'actual_sleep_seconds'))
        require(all(number(v) for v in (work, requested, actual)), 'numbers')
        reset = row['frame'] in (FIRST, start)
        require(row['baseline_only'] is reset and (not reset or work == requested == 0), 'baseline')
        require(requested == min(work, MAX_SLEEP_SECONDS) and actual >= requested, 'duration')
        require(row['sleep_completed'] is True, 'sleep_completed')
    return dict(pacing_rows=len(rows), actual_sleep_seconds=sum(r['actual_sleep_seconds'] for r in rows),
                requested_sleep_seconds=sum(r['requested_sleep_seconds'] for r in rows), thermal_safety_certified=False)
