"""保存観測を実Basis・元配置・同一原Jへ結合。構造のみの検査と分離する。"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from typing import Any


def equal(left: Any, right: Any) -> None:
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True), 'private_samecall_value'


def journal_step(rows: Any, frame: int, scope: Any) -> Any:
    found = [row for row in rows if row['kind'] == 'step' and row['frame_idx'] == frame
        and row['side'] == scope[-1]]
    assert len(found) == 1
    row = found[0]
    assert row['status'] == 'returned' and row['exception'] is None
    equal([row['source_id'], row['run_id'], row['software_reset'], row['pipe_object_id'],
        row['generation']['reset_epoch'], row['side']], [scope[i] for i in (0, 1, 2, 3, 5, 6)])
    assert row['time_sec'] == frame/60
    return row


def verify(kept: Any, evidence: Any, saved: Any, journal: Any, history: Any) -> dict[str, Any]:
    factory, state = kept['factory'], kept['state']
    control, basis = factory.controller, kept['basis']
    assert control.provider is factory.provider and not control.calls and not control.tickets
    values = evidence.verify(control)
    equal(saved, values)
    assert [v['kind'] for v in values] == ['private_suffix_basis_capture/v1', 'private_suffix_start_return/v1']
    payloads = [json.loads(v['payload_json']) for v in values]
    first, start = payloads
    binding = control.history[first['scope'][-1]]
    actual = basis.validate(control, binding, binding.private_suffix_basis)
    proof = kept['placement'].committed(basis, control, binding)
    expected = evidence.source(binding, actual)
    for value in payloads:
        equal({key: value[key] for key in expected}, expected)
    assert first['consumed'] is True and first['call_frame'] == values[0]['frame'] == actual.frame
    equal(first['post_queue_ref_ids'], [id(actual.head)])
    equal(start['started'], actual.started)
    equal(start['started_state'], asdict(actual.started_state))
    equal(start['scope_at_start'], actual.scope)
    equal(start['refs_at_start'], [id(actual.head)])
    assert start['queue_at_start'] == id(actual.queue)
    equal(start['tokens'], [actual.token])
    assert start['added'] == [] and start['quiet'] is True and start['original_return'] is True
    equal(start['next_pair'], actual.source.proof['new_accepted'])
    equal(start['dnext_pair'], actual.source.proof['dnext'])
    assert actual.frame < start['frame_at_start'] == values[1]['frame'] == actual.started[0]
    assert start['clock_at_start'] == actual.started[1] and len(values) == 2
    for frame in (actual.frame, actual.started[0], proof['available_frame']):
        journal_step(journal, frame, actual.scope)
    at_start = [r for r in history if r['scope']['frame_idx'] == actual.started[0]
        and r['scope']['side'] == actual.scope[-1]]
    assert len(at_start) == 1
    equal(at_start[0]['decision']['history_state'], asdict(actual.started_state))
    assert start['started_state']['clock']['sequence'] == actual.started[0]*4+2
    return dict(private_samecall_live_verified=True, basis_frame=actual.frame,
        start_frame=actual.started[0], records=len(values), original_J_joined=True,
        retrospective_quiet_fabricated=False, physical_certified=False, quality_gate_clear=False)
