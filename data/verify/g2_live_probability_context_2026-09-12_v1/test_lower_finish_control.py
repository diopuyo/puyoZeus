"""新driverの結果分離/元例外/フック復元。下位品質本体の合格ではない。"""
from __future__ import annotations
import ast
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent


@pytest.mark.parametrize('fails', [False, True])
def test_driver_preserves_original_and_restores(tmp_path: Path, fails: bool) -> None:
    path = ROOT / 'probe_lower_finish.py'
    tree = ast.parse(path.read_bytes())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls: list[str] = []
    kept = dict(state=dict(output=tmp_path))
    old_result = dict(old=True)
    error = RuntimeError('actual_lower_failure')
    def prior(value: Any) -> Any:
        assert value is kept
        calls.append('original')
        return old_result
    def lower(value: Any) -> Any:
        assert value is kept
        calls.append('lower')
        if fails:
            raise error
        return dict(full_probability_finalizer_verified=False, quality_gate_clear=False)
    old = N(OLD=N(closed_samecall=prior), driver=object(), SOURCES=('old.py',))
    def delegated() -> None:
        assert old.OLD.closed_samecall(kept) is old_result
    old.main = delegated
    before = old.driver, old.SOURCES
    namespace = dict(OLD=old, lower=lower, driver=object(), Any=Any, Path=Path,
                     ExitStack=ExitStack, json=json, RESULT='PROBABILITY_LOWER_FINISH.json')
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    if fails:
        with pytest.raises(RuntimeError) as caught:
            namespace['main']()
        assert caught.value is error
        value = json.loads((tmp_path / 'PROBABILITY_LOWER_FAILURE.json').read_bytes())
        assert value['error'] == repr(error)
        assert value['owner_result'] == old_result
        assert not (tmp_path / namespace['RESULT']).exists()
    else:
        namespace['main']()
        value = json.loads((tmp_path / namespace['RESULT']).read_bytes())
        assert value['full_probability_finalizer_verified'] is False
    assert calls == ['original', 'lower']
    assert old.OLD.closed_samecall is prior and (old.driver, old.SOURCES) == before
