"""補助観測の保存失敗と構造検査を分離する。構造本体はstubと明記。"""
from __future__ import annotations
import ast
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest


@pytest.mark.parametrize('exists', [False, True])
def test_observation_failure_does_not_skip_structure(tmp_path: Path, exists: bool) -> None:
    path = Path(__file__).resolve().parent / 'probe_retired_stage.py'
    fn = next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    output = tmp_path / 'OUTER_PROBABILITY_FACTS.json'
    if exists:
        output.write_text('old')
    factory = N(controller=N(history={}))
    guard = N(frame=100, error=None, record=None, binding=None)
    guard.reset_lease = N(guard=guard)
    state = dict(output=tmp_path, repeat_scope_guard=guard)
    state['probabilistic_tracking_mode'] = N(state=state, native=N(last_frame=100))
    state['probabilistic_basis_connection'] = N(recovery=N(factory=factory))
    kept = dict(state=state, factory=factory)
    calls, results = [], []
    def prior(value: Any) -> Any:
        assert value is kept
        calls.append('stage_stub')
        return dict(stage1_verified=True, quality_gate_clear=False)
    old = N(STAGE=N(verify=prior), OLD=N(OLD=N(SOURCES=('a.py',))), driver=object())
    old.main = lambda: results.append(old.STAGE.verify(kept))
    before = old.driver, old.OLD.OLD.SOURCES
    ns = dict(OLD=old, driver=object(), Any=Any, ExitStack=ExitStack, json=json, sys=sys)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    ns['main']()
    assert calls == ['stage_stub'] and old.STAGE.verify is prior
    assert (old.driver, old.OLD.OLD.SOURCES) == before
    assert results[0]['outer_facts_saved'] is (not exists)
    assert (results[0]['outer_facts_error'] is None) is (not exists)
    if exists:
        assert output.read_text() == 'old'
    else:
        assert json.loads(output.read_text())['actual_video'] is False
