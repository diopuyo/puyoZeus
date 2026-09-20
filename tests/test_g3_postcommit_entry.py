"""同じ所有scopeの重複Bridgeを、元constructorを呼ぶ前に拒否する。"""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import sys
import pytest
from scripts import g3_postcommit_entry as E


def test_duplicate_bridge_does_not_call_original(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[Any] = []
    def original(*args: Any, **kwargs: Any) -> object:
        calls.append(args)
        return object()
    owner = N(Bridge=original)
    def replace(stack: Any, module: Any, name: str, value: Any) -> None:
        stack.callback(setattr, module, name, getattr(module, name))
        setattr(module, name, value)
    adapter = N(A=N(A=N(A=N(V4=N(A=N(W=owner), replace_owned=replace)))))
    class Consumer:
        pass
    monkeypatch.setitem(sys.modules, Consumer.__module__, sys.modules[__name__])
    monkeypatch.setattr(E.B, 'bind', lambda *args, **kwargs: N(path=tmp_path / 'rows'))
    monkeypatch.setattr(E.F, 'install', lambda *args: {})
    monkeypatch.setattr(E.G, 'save', lambda *args: None)
    state = dict(postcommit_publication_consumer=Consumer(), output=tmp_path)
    with ExitStack() as stack:
        E.install(stack, adapter)
        owner.Bridge(None, state, None)
        with pytest.raises(ValueError, match='duplicate_bridge'):
            owner.Bridge(None, state, None)
        assert len(calls) == 1
    assert owner.Bridge is original
