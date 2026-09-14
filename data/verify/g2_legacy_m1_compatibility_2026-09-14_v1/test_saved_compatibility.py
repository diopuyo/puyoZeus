"""新保存consumerの未評価票検査だけを人工保存対照で検査する。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'g2_second_prefix_runtime_2026-09-14_v38'))
import review_m1 as R


def fixture_values() -> dict:
    state = dict(last=R.W.LAST, accepted=[], pending=None)
    compat = dict(status='unsupported', old_finish_error=R.A.COMPAT.C.INCOMPLETE,
        legacy_m1_complete=False, observation_reached_end=True, quality_gate_clear=False, schedule=state)
    session = dict(schedule=state, saved=[], error=None, session_error=None, restored=True,
        observer_closed=True, witness_closed=True, evaluation_flags_closed=True)
    child = dict(closed=True, child_exit_code=0, accepted_count=0)
    return {'LEGACY_M1_COMPATIBILITY.json':compat, 'BELIEF_M1_SESSION.json':session,
        'JOINT_EVENTS.jsonl.complete.json':child}


@pytest.mark.parametrize('case', ['valid', 'pending', 'error', 'unclosed', 'child_exit', 'count', 'hold'])
def test_saved_unsupported_is_not_quality_pass(tmp_path: Path, case: str) -> None:
    values = fixture_values()
    compat, session, child = values.values()
    if case == 'pending': compat['schedule']['pending'] = R.W.LAST
    if case == 'error': session['session_error'] = 'real_failure'
    if case == 'unclosed': session['restored'] = False
    if case == 'child_exit': child['child_exit_code'] = 1
    if case == 'count': child['accepted_count'] = 1
    for name, value in values.items():
        (tmp_path/name).write_text(json.dumps(value), encoding='utf-8')
    reason = R.A.COMPAT.C.REASON
    decision = dict(frame=R.W.LAST, reasons=[reason], action='WAIT', saved=False)
    (tmp_path/'M1_CAPTURE_SCHEDULE.jsonl').write_text(json.dumps(decision)+'\n', encoding='utf-8')
    hold = dict(frame=R.W.LAST, reason=reason, requested=case == 'hold')
    (tmp_path/'LEGACY_M1_COMPATIBILITY.jsonl').write_text(json.dumps(hold)+'\n', encoding='utf-8')
    if case == 'valid':
        result = R.unsupported_saved(tmp_path)
        assert result['legacy_M1_complete'] is False and result['quality_gate_clear'] is False
        assert result['actual_saved_count'] == 0 and result['explicit_raw_stable_holds'] == 1
    else:
        with pytest.raises(ValueError): R.unsupported_saved(tmp_path)
