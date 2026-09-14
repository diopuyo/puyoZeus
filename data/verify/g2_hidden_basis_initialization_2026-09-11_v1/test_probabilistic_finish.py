"""実v34保存票の整合と破損拒否。生factory成功の代替にはしない。"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
import probabilistic_finish as F

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v34'
sys.path.insert(0, str(ROOT.parent / 'g2_probabilistic_scope_candidate_2026-09-11_v1'))
import serialization as S


def test_saved_actual_basis_and_inputs_are_consistent() -> None:
    value, report = F.saved(OUTPUT, S)
    assert value.frame == 34944 and len(value.worlds) == 2401
    assert report['saved_joint_decoded'] and not report['live_factory_verified']
    assert not report['runtime_finalization_allowed'] and not report['quality_gate_clear']


@pytest.mark.parametrize('fault', ['body', 'empty_target', 'subset', 'permission', 'basis_token', 'scope', 'native', 'input', 'restore'])
def test_changed_saved_evidence_is_rejected(fault: str, monkeypatch: Any) -> None:
    original_read, original_lines = F.read, F.lines
    docs = {name: original_read(OUTPUT, name) for name in ('RESULT.json', 'PROBABILISTIC_TARGET_RESULT.json',
        'SETTLED_BASIS_OBSERVER.json', 'QUALIFIED_PRIOR_VOTES.json', 'PROBABILISTIC_TRACKING_STATUS.json',
        'INFLIGHT_QUARANTINE.json')}
    logs = {name: original_lines(OUTPUT, name) for name in ('PROBABILISTIC_BASIS.jsonl',
        'PROBABILISTIC_TRACKING.jsonl', 'BASIS_ONLY_INPUT.jsonl')}
    if fault == 'body': docs['RESULT.json']['exit_code'] = 1
    if fault == 'empty_target': docs['PROBABILISTIC_TARGET_RESULT.json'] = {}
    if fault == 'subset': docs['PROBABILISTIC_TARGET_RESULT.json']['basis_only'] = False
    if fault == 'permission': logs['PROBABILISTIC_BASIS.jsonl'][0]['state']['integer_current_permission'] = True
    if fault == 'basis_token': logs['PROBABILISTIC_BASIS.jsonl'][0]['source_call_token'] = 'foreign'
    if fault == 'scope': logs['PROBABILISTIC_BASIS.jsonl'][0]['state']['scope'][1] = 'other-run'
    if fault == 'native': docs['PROBABILISTIC_TRACKING_STATUS.json']['pending_native_consumptions'] = 1
    if fault == 'input': logs['BASIS_ONLY_INPUT.jsonl'][0]['frame'] -= 2
    if fault == 'restore': docs['INFLIGHT_QUARANTINE.json']['references_restored'] = False
    monkeypatch.setattr(F, 'read', lambda output, name: deepcopy(docs[name]))
    monkeypatch.setattr(F, 'lines', lambda output, name: deepcopy(logs[name]))
    with pytest.raises((RuntimeError, ValueError)):
        F.saved(OUTPUT, S)


def test_nonprobabilistic_finish_falls_back_without_rewrite() -> None:
    calls: list[Any] = []
    module = N(__file__=str(F.EXPECTED_FINISH), finish=lambda kept: calls.append(kept))
    loader = F.wrap(lambda original: original)(lambda *_: module)
    loaded = loader('_empty_fused_finish', F.EXPECTED_FINISH, None)
    kept = dict(state={})
    loaded.finish(kept)
    assert calls == [kept]
    assert loader('unrelated_alias', F.EXPECTED_FINISH, None) is module


def test_probabilistic_dispatch_uses_live_checker(monkeypatch: Any) -> None:
    module = N(__file__=str(F.EXPECTED_FINISH), finish=lambda kept: pytest.fail('旧整数へ流さない'))
    loader = F.wrap(lambda original: original)(lambda *_: module)
    calls: list[Any] = []
    report = dict(fake_test_result=True, quality_gate_clear=False)
    monkeypatch.setattr(F, 'live', lambda kept: report)
    kept = dict(state=dict(probabilistic_tracking_mode=object(), output=OUTPUT),
                runtime=N(write=lambda path, value: calls.append((path.name, value))))
    loader('_empty_fused_finish', F.EXPECTED_FINISH, None).finish(kept)
    assert calls == [('PROBABILISTIC_CPU_FINISH.json', report)]
