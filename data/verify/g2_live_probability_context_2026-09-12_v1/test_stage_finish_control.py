"""構造driverの先行結果保持/例外同一性/フック復元。構造本体は別検査。"""
from __future__ import annotations
import ast
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest


@pytest.mark.parametrize('fails', [False, True])
def test_stage_driver_control(tmp_path: Path, fails: bool) -> None:
    path = Path(__file__).resolve().parent / 'probe_stage_finish.py'
    fn = next(n for n in ast.parse(path.read_bytes()).body
              if isinstance(n, ast.FunctionDef) and n.name == 'main')
    kept = dict(state=dict(output=tmp_path))
    calls: list[str] = []
    prior_result = dict(world_pass=True)
    failure = RuntimeError('actual_structure_failure')
    def prior(value: Any) -> Any:
        assert value is kept
        calls.append('lower')
        return prior_result
    def verify(value: Any) -> Any:
        assert value is kept
        calls.append('stage')
        if fails:
            raise failure
        return dict(stage1_verified=True, full_probability_finalizer_verified=False)
    old = N(OLD=N(SOURCES=('old.py',)), driver=object(), lower=prior)
    def delegated() -> None:
        assert old.lower(kept) is prior_result
    old.main = delegated
    before = old.driver, old.OLD.SOURCES
    ns = dict(OLD=old, STAGE=N(verify=verify), driver=object(), Path=Path,
              Any=Any, ExitStack=ExitStack, json=json)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    if fails:
        with pytest.raises(RuntimeError) as caught:
            ns['main']()
        assert caught.value is failure
        value = json.loads((tmp_path / 'PROBABILITY_STAGE_FAILURE.json').read_bytes())
        assert value['lower_result'] == prior_result and value['error'] == repr(failure)
        assert not (tmp_path / 'PROBABILITY_STAGE_FINISH.json').exists()
    else:
        ns['main']()
        value = json.loads((tmp_path / 'PROBABILITY_STAGE_FINISH.json').read_bytes())
        assert value['full_probability_finalizer_verified'] is False
    assert calls == ['lower', 'stage']
    assert old.lower is prior and (old.driver, old.OLD.SOURCES) == before
