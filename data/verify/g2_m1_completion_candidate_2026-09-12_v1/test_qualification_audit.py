"""同一root保存票の結合を人工全列で検査。実画像/物理分布の証明ではない。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import check_saved_inputs
import qualification_audit as A
from test_schedule_saved import test_saved_payload_and_sha as create_saved


def write(path: Path, values: list[dict]) -> None:
    path.write_text('\n'.join(json.dumps(v) for v in values), encoding='utf-8')


def observation(frame: int, side: str) -> tuple[dict, dict]:
    scope = dict(source_id='source', run_id='run', frame_idx=frame, time_sec=frame / 60,
                 side=side, pipe_object_id=1, generation=dict(reset_epoch=0, action_revision=0))
    step = scope | dict(kind='step', token=f'{side}:{frame}', software_reset=0, events=[], status='returned',
        exception=None, generation_after=scope['generation'], returned=dict(state='STABLE', active_origin=None))
    flag = dict(scope=scope, token=step['token'], software_reset=0, hold_reason=None, state='stable',
                origin_present=False, grace_end=None, match_active=True, effect_window=frame == 35370 and side == '2P')
    return step, flag


def context(frame: int) -> dict:
    sides = {side: dict(pb={}, sm={}, next={}, candidate_row={}, identity={}, hold_reasons=[],
        before_hold=dict(state='STABLE', probability=dict(present=True, type_valid=True, errors=[]))) for side in A.Q.SIDES}
    update = {}
    for key, value in (('is_match_active', True), ('match_end_locked', False), ('post_match_lockdown_active', False)):
        update.update({key: value, key + '_observed': True})
    return dict(source_id='source', run_id='run', frame_idx=frame, time_sec=frame / 60, capture_status='CAPTURED',
                failures=[], upstream_failures={}, sides=sides, update=update, hold_reasons=[])


def fixture(root: Path) -> None:
    create_saved(root, 'none')
    schedule = list(A.rows(root / 'M1_CAPTURE_SCHEDULE.jsonl'))
    journal, contexts, first = [], [], []
    for row in schedule:
        frame = row['frame']
        pair = [observation(frame, side) for side in A.Q.SIDES]
        journal.extend(p[0] for p in pair)
        row['flags'] = dict(zip(A.Q.SIDES, (p[1] for p in pair)))
        row['reasons'] = ['before_window'] if frame < 35370 else ['2P:effect_window'] if frame == 35370 else []
        contexts.append(context(frame))
        first.append(dict(scope=pair[0][1]['scope'], journal_token=pair[0][0]['token'], pending_occurrences=[],
                          arrival_ledger=dict(arrivals=[], applied=[], acknowledgements=[])))
    status = json.loads((root / 'BELIEF_M1_SESSION.json').read_text())
    status.update({k: None for k in ('error', 'session_error', 'observer_error', 'witness_error', 'evaluation_flags_error')})
    status['modes'] = [dict(initial=dict(state=dict(scope=['source', 'run', 0, 1, 2, 0, '2P'], frame=35368),
        source_call_token='2P:35368'), retired=None, applied=[], error=None, closed=True)]
    (root / 'BELIEF_M1_SESSION.json').write_text(json.dumps(status))
    for name, values in (('M1_CAPTURE_SCHEDULE.jsonl', schedule), ('atomic_journal.jsonl', journal),
                         ('provisional_context.jsonl', contexts), ('PROBABILISTIC_TRACKING.jsonl', first)):
        write(root / name, values)


@pytest.mark.parametrize('case', ['normal', 'wrong_reason', 'other_run'])
def test_complete_saved_root(tmp_path: Any, case: str) -> None:
    fixture(tmp_path)
    if case == 'wrong_reason':
        values = list(A.rows(tmp_path / 'M1_CAPTURE_SCHEDULE.jsonl'))
        values[1]['reasons'] = ['1P:not_stable']
        write(tmp_path / 'M1_CAPTURE_SCHEDULE.jsonl', values)
    elif case == 'other_run':
        values = list(A.rows(tmp_path / 'provisional_context.jsonl'))
        values[1]['run_id'] = 'other'
        write(tmp_path / 'provisional_context.jsonl', values)
    if case == 'normal':
        result = A.verify(tmp_path)
        assert result['saved_reasons_reconstructed'] and result['compared_frames'] == 465
        assert not result['quality_gate_clear'] and not result['physical_replay_proven']
    else:
        with pytest.raises(ValueError): A.verify(tmp_path)
