"""終了wrapper実装の装着/非対象委譲/alias解除。実factoryは生成しない。"""
from contextlib import ExitStack
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
import live_adapter as A


def target() -> Any:
    FINISH = N(__file__=str(A.PRIOR / 'live_finish_v2.py'))
    def evaluate(*args: Any) -> Any:
        return FINISH
    return N(evaluate=evaluate)


def test_real_install_and_nonreset_restore(tmp_path: Any, monkeypatch: Any) -> None:
    for name in A.FINISH_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    selected = target()
    previous = selected.evaluate
    before = list(sys.path)
    main = N(__globals__=dict(Q=N(FINAL=selected), K=N(write=lambda *args: pytest.fail('非対象保存'))))
    with ExitStack() as stack:
        A.install_finish(stack, main)
        assert sys.path == before and all(name in sys.modules for name in A.FINISH_MODULES)
        assert selected.evaluate(None, [], None, tmp_path, {}) is previous()
    assert selected.evaluate is previous and sys.path == before
    assert all(name not in sys.modules for name in A.FINISH_MODULES)


def test_foreign_alias_rejected(monkeypatch: Any) -> None:
    for name in A.FINISH_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, A.FINISH_MODULES[0], object())
    selected = target()
    main = N(__globals__=dict(Q=N(FINAL=selected)))
    with ExitStack() as stack, pytest.raises(AssertionError, match='foreign_alias'):
        A.install_finish(stack, main)
