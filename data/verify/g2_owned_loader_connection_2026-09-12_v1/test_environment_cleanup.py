"""原sessionの終了時module削除と、新alias所有者の解除を組み合わせる。"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
import importlib.util
from types import FunctionType, SimpleNamespace as N
from typing import Any, Iterator
import sys
import pytest
import owned_adapter as R
from test_owned_lifetime import fixture


def scope() -> Any:
    # 原configuredのcleanup codeは変えず、今回無関係なモデル生成だけを人工化する。
    with ExitStack() as stack:
        entry = R.configured(stack)
        session = entry.__globals__['S']
        original = session.configured.__wrapped__
    @contextmanager
    def installed() -> Iterator[dict]:
        yield {}
    repeated = N(load=lambda stack: None)
    cloned = FunctionType(original.__code__, dict(original.__globals__, REPEAT=repeated, installed=installed))
    return contextmanager(cloned)


def test_original_environment_cleanup_then_alias_release(monkeypatch: Any) -> None:
    cleanup = scope()
    owner, original, *_ = fixture(monkeypatch)
    with ExitStack() as stack:
        R.install_side(stack, owner)
        with cleanup():
            owner.dependencies()
            assert R.SIDE_ALIAS in sys.modules
        assert R.SIDE_ALIAS not in sys.modules, 'original_environment_did_not_remove_alias'
    assert owner.dependencies is original


def test_active_alias_disappearance_is_rejected(monkeypatch: Any) -> None:
    owner, *_ = fixture(monkeypatch)
    with ExitStack() as stack:
        R.install_side(stack, owner)
        owner.dependencies()
        del sys.modules[R.SIDE_ALIAS]
        with pytest.raises(RuntimeError, match='repair_side_replaced'):
            owner.dependencies()
