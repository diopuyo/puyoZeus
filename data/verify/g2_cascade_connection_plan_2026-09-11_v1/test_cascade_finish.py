"""人工保存票で新finisherの結合条件を検査する。実Jの成功票ではない。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / n) for n in ('g2_hidden_basis_initialization_2026-09-11_v1',
    'g2_archive_close_boundary_2026-09-11_v1', 'g2_probabilistic_scope_candidate_2026-09-11_v1')]
import cascade_finish as F
import serialization as S
import belief as B

OUTPUT = ROOT / 'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v39'


def artifacts() -> tuple[Any, Any]:
    docs = {n: F.P.read(OUTPUT, n) for n in ('RESULT.json', 'PROBABILISTIC_TARGET_RESULT.json',
        'SETTLED_BASIS_OBSERVER.json', 'QUALIFIED_PRIOR_VOTES.json',
        'PROBABILISTIC_TRACKING_STATUS.json', 'INFLIGHT_QUARANTINE.json')}
    logs = {n: F.P.lines(OUTPUT, n) for n in ('PROBABILISTIC_BASIS.jsonl',
        'PROBABILISTIC_TRACKING.jsonl', 'BASIS_ONLY_INPUT.jsonl')}
    target = docs['PROBABILISTIC_TARGET_RESULT.json']
    target.update(basis_only=False, basis_cascade_only=True, actual_J_steps=F.V.J_STEPS, physical_transitions=1)
    docs['RESULT.json'].update(updates=F.V.UPDATES, J_steps=F.V.J_STEPS)
    docs['RESULT.json']['reset_continuation'].update(deepcopy(target))
    initial = S.decode(logs['PROBABILISTIC_BASIS.jsonl'][0]['state'])
    start, template = logs['BASIS_ONLY_INPUT.jsonl'][0]['frame'], logs['BASIS_ONLY_INPUT.jsonl'][0]
    logs['BASIS_ONLY_INPUT.jsonl'] = [deepcopy(template) | dict(frame=start + i*F.V.I.STRIDE)
                                    for i in range(F.V.I.POST_COUNT)]
    template = logs['PROBABILISTIC_TRACKING.jsonl'][0]
    rows = []
    for frame in range(initial.frame+F.V.I.STRIDE, logs['BASIS_ONLY_INPUT.jsonl'][-1]['frame']+F.V.I.STRIDE, F.V.I.STRIDE):
        row = deepcopy(template)
        row['scope']['frame_idx'] = frame
        row['scope']['time_sec'] = frame / 60
        row['journal_token'] = 'step:' + str(frame - 34772)
        rows.append(row)
    logs['PROBABILISTIC_TRACKING.jsonl'] = rows
    docs['PROBABILISTIC_TRACKING_STATUS.json']['rows'] = len(rows)
    receipt = transition(initial, rows[-1]['scope']['frame_idx'], rows[-1]['journal_token'])
    rows[-1].update(reason='basis_cascade_applied', transition=receipt)
    docs['BASIS_CASCADE_FINAL.json'] = dict(transition=receipt, active_origin=False,
        final_state='stable', final_time=rows[-1]['scope']['frame_idx']/60, chain_until=0.0)
    docs['BASELINE_RETIREMENT.json'] = dict(used=True, tracker_restored=True,
        rows=[dict(suppressed=True, settlement_token=receipt['source_call_token'])])
    logs['atomic_journal.jsonl'] = [dict(kind='step', frame_idx=frame, side=initial.scope[-1],
        token=token, status='returned', exception=None, source_id=initial.scope[0], run_id=initial.scope[1],
        software_reset=initial.scope[2], pipe_object_id=initial.scope[3])
        for frame, token in ((receipt['origin']['first_observed_frame'], receipt['origin']['source_call_token']),
                             (receipt['applied_frame'], receipt['source_call_token']))]
    return docs, logs


def transition(initial: Any, frame: int, token: str) -> Any:
    path = ROOT / 'g2_basis_cascade_candidate_2026-09-11_v1/cascade.py'
    spec = importlib.util.spec_from_file_location('_g2_finish_test_cascade', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    origin = B.Board.from_dict({'grid': initial.worlds[0].grid})
    observed = B.ChainSimulator(exclude_hidden_row_from_pop=True).simulate(origin).final_board
    after, report = module.settle(initial, initial.scope, frame, token, origin, observed)
    return dict(kind='basis_cascade', next_consumed=False, physical_certified=False,
        applied_frame=frame, source_call_token=token, state=S.encode(after), distribution_report=asdict(report),
        origin=dict(first_observed_frame=initial.frame+2, source_call_token='step:' + str(initial.frame+2-34772)))


@pytest.mark.parametrize('fault', ['none', 'gap', 'token', 'permission', 'extra_transition', 'active', 'native', 'body', 'original_j'])
def test_synthetic_saved_contract(fault: str, monkeypatch: Any) -> None:
    docs, logs = artifacts()
    rows = logs['PROBABILISTIC_TRACKING.jsonl']
    if fault == 'gap': rows.pop(0)
    if fault == 'token': rows[-1]['journal_token'] = 'foreign'
    if fault == 'permission': docs['BASIS_CASCADE_FINAL.json']['transition']['state']['integer_current_permission'] = True
    if fault == 'extra_transition': rows[-2].update(reason='basis_cascade_applied', transition=rows[-1]['transition'])
    if fault == 'active': docs['BASIS_CASCADE_FINAL.json']['active_origin'] = True
    if fault == 'native': docs['PROBABILISTIC_TRACKING_STATUS.json']['pending_native_consumptions'] = 1
    if fault == 'body': docs['RESULT.json']['exit_code'] = 1
    if fault == 'original_j': logs['atomic_journal.jsonl'][-1]['token'] = 'foreign'
    monkeypatch.setattr(F.P, 'read', lambda _output, name: deepcopy(docs[name]))
    monkeypatch.setattr(F.P, 'lines', lambda _output, name: deepcopy(logs[name]))
    if fault == 'none':
        _, report = F.saved(OUTPUT, S)
        assert report['saved_joint_decoded'] and not report['runtime_finalization_allowed']
    else:
        with pytest.raises((ValueError, RuntimeError)):
            F.saved(OUTPUT, S)


@pytest.mark.parametrize('token', ['step:2', 'synthetic-test-step:12'])
def test_issued_token_is_original_J_ordinal(token: str) -> None:
    journal = N(expected=[(10, '1P'), (10, '2P'), (12, '1P')], steps=3)
    receipt = dict(applied_frame=12, source_call_token=token,
                   origin=dict(first_observed_frame=10, source_call_token='step:0'))
    if token == 'step:2': F.audit_issued(journal, receipt, '1P')
    else:
        with pytest.raises(RuntimeError, match='cascade_live_issued_token'):
            F.audit_issued(journal, receipt, '1P')
