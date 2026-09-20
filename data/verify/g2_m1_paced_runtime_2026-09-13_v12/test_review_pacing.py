"""全列と開始境界の人工保存票で休止検収の拒否条件を確認。"""
from __future__ import annotations
import json
from typing import Any
import pytest
import review_pacing as R


@pytest.mark.parametrize('case', ['normal', 'missing', 'duration', 'baseline', 'incomplete', 'error'])
def test_saved_pacing(tmp_path: Any, case: str) -> None:
    start = R.FIRST + R.STRIDE
    rows = [dict(frame=f, baseline_only=f in (R.FIRST, start), work_seconds=0 if f <= start else .25,
                 requested_sleep_seconds=0 if f <= start else .25, actual_sleep_seconds=.3,
                 sleep_completed=True) for f in range(R.FIRST, R.LAST, R.STRIDE)]
    value = dict(rows=rows, closed=True, error=None, body_error=None,
                 quality_gate_clear=False, max_sleep_seconds=1.0)
    if case == 'missing': rows.pop()
    if case == 'duration': rows[-1]['actual_sleep_seconds'] = .1
    if case == 'baseline': rows[-1]['baseline_only'] = True
    if case == 'incomplete': rows[-1]['sleep_completed'] = False
    if case == 'error': value['error'] = 'original_sleep_failure'
    (tmp_path / 'FRAME_PACING_STATUS.json').write_text(json.dumps(value))
    (tmp_path / 'WHOLE_M1_DRIVER_STATUS.json').write_text(json.dumps(dict(start_frame=start)))
    if case == 'normal': assert R.verify(tmp_path)['pacing_rows'] == len(rows)
    else:
        with pytest.raises(ValueError): R.verify(tmp_path)
