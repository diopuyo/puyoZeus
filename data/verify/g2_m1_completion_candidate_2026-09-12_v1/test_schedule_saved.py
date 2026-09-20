"""独立保存順序検査の人工対照。資格フラグ/物理真値を証明しない。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

import pytest
import schedule_saved as V


def rows() -> list[dict]:
    result = []
    for frame in range(35368, V.END + 1, V.STRIDE):
        due = 35370 if frame <= 35372 else 35412 if frame <= 35412 else None
        action = 'REQUEST' if frame in (35372, 35412) else 'COMPLETE' if due is None else 'WAIT'
        result.append(dict(frame=frame, reasons=['not_stable'] if frame <= 35370 else [],
            due=due, action=action, saved=action == 'REQUEST', original_sentinel=frame in V.EARLIEST,
            quality_gate_clear=False))
    return result


@pytest.mark.parametrize('case', ['normal', 'gap', 'missing', 'ack', 'late', 'sentinel', 'end'])
def test_order(case: str) -> None:
    value = deepcopy(rows())
    if case == 'gap': value[1]['frame'] += 2
    elif case == 'missing': value.pop(10)
    elif case == 'ack': value[2]['saved'] = False
    elif case == 'late': value[2]['action'] = 'WAIT'
    elif case == 'sentinel': value[1]['original_sentinel'] = False
    elif case == 'end': value.pop()
    if case == 'normal': assert V.order(value) == (35372, 35412)
    else:
        with pytest.raises(ValueError): V.order(value)


@pytest.mark.parametrize('tamper', ['none', 'sha', 'prefix'])
def test_saved_payload_and_sha(tmp_path: Any, tamper: str) -> None:
    values, saved = rows(), []
    for frame in (35372, 35412):
        path = tmp_path / f'BELIEF_M1_{frame}.json'
        raw = json.dumps(dict(result=dict(frame=frame), parent_recapture_verified=True)).encode()
        path.write_bytes(raw)
        saved.append(dict(frame=frame, path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
    (tmp_path / 'M1_CAPTURE_SCHEDULE.jsonl').write_text('\n'.join(map(json.dumps, values)))
    status = dict(original_targets=list(V.EARLIEST), schedule_rows=len(values), saved=saved,
                  schedule=dict(last=V.END, accepted=[35372, 35412], pending=None))
    (tmp_path / 'BELIEF_M1_SESSION.json').write_text(json.dumps(status))
    attach = dict(start_frame=35366 if tamper != 'prefix' else 35364, last_frame=V.END, installed=True,
                  closed=True, stopped=True, consumer_forbidden=True, session_restored=True,
                  error=None, body_error=None, quality_gate_clear=False)
    (tmp_path / 'WHOLE_M1_DRIVER_STATUS.json').write_text(json.dumps(attach))
    if tamper == 'sha':
        path.write_bytes(b'{}')
        with pytest.raises(ValueError, match='saved_sha'): V.verify(tmp_path)
    elif tamper == 'prefix':
        with pytest.raises(ValueError, match='attach_prefix'): V.verify(tmp_path)
    else:
        report = V.verify(tmp_path)
        assert report['order_and_saved_files_verified'] and not report['qualification_proven']
