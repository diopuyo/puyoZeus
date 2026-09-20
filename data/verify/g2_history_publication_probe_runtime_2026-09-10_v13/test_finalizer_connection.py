"""同runの原票をそのまま渡し、失敗は巻き戻さず伝播させる。"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import json
import sys
from typing import Any
import pytest
import finalizer_connection as F


def inputs(tmp_path: Path) -> tuple[Any, Any, Any, Any]:
    with (tmp_path/'atomic_journal.jsonl').open('x') as stream:
        json.dump({'kind':'step','token':'actual'},stream)
    firing = [{'stage':'test-only'}]
    state = {'repeated_firing_constructor':dict(installed=True, closed=True,
        references_restored=True, rows=firing)}
    return object(), [], object(), state


def test_actual_evidence_forwarded(tmp_path: Path, monkeypatch: Any) -> None:
    goals, rows, legal, state = inputs(tmp_path)
    paths, result, calls = list(sys.path), {'actual_result':True}, []
    def evaluate(g: Any, r: Any, l: Any, o: Any, **kwargs: Any) -> Any:
        assert g is goals and r is rows and l is legal and o == tmp_path
        assert kwargs['firing_rows'] is state['repeated_firing_constructor']['rows']
        assert kwargs['journal_rows'] == [{'kind':'step','token':'actual'}]
        calls.append(True)
        return result
    def load(alias: str, path: Path, stack: Any) -> Any:
        assert alias == '_live_repeated_finalizer' and path == F.ROOT/'compat.py'
        assert sys.path[0] == str(F.ROOT)
        return SimpleNamespace(evaluate=evaluate)
    monkeypatch.setattr(F.K,'load',load)
    assert F.evaluate(goals,rows,legal,tmp_path,state) is result
    assert calls == [True] and sys.path == paths


@pytest.mark.parametrize('field', ('installed','closed','references_restored'))
def test_unclosed_constructor_rejected(tmp_path: Path, field: str) -> None:
    goals, rows, legal, state = inputs(tmp_path)
    state['repeated_firing_constructor'][field] = False
    with pytest.raises(ValueError, match='finalizer_constructor_unclosed'):
        F.evaluate(goals,rows,legal,tmp_path,state)


def test_audit_error_restores_scope(tmp_path: Path, monkeypatch: Any) -> None:
    goals, rows, legal, state = inputs(tmp_path)
    before, error = list(sys.path), ValueError('actual_audit_failure')
    def evaluate(*args: Any, **kwargs: Any) -> Any:
        raise error
    def load(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(evaluate=evaluate)
    monkeypatch.setattr(F.K,'load',load)
    with pytest.raises(ValueError) as caught:
        F.evaluate(goals,rows,legal,tmp_path,state)
    assert caught.value is error and sys.path == before and not (tmp_path/'COMPLETE').exists()
