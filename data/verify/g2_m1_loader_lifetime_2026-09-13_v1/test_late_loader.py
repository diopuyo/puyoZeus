"""遅延参照の所有権/寿命を限定負例で検査する。実Session対照は別probe。"""
from contextlib import ExitStack
from types import SimpleNamespace as N
from typing import Any
import sys

import pytest
import late_loader as L


@pytest.mark.parametrize('case', ['normal', 'closed', 'foreign_module', 'foreign_hook',
    'initial_owner', 'same_source_replaced', 'hook_released'])
def test_loader_lifetime(monkeypatch: Any, case: str) -> None:
    """元委譲を維持し、解放済み/他所有参照を通さない。"""
    original = lambda alias, path, injection=None: (alias, path, injection)
    bootstrap = N(__name__='_limited_late_loader_test', load=original)
    monkeypatch.setitem(sys.modules, bootstrap.__name__, bootstrap)
    owner = N(bootstrap=lambda: bootstrap)
    creator = lambda load, frames: load
    dependencies = N(session_creator=creator)

    def replace(stack: Any, module: Any, key: str, value: Any) -> None:
        prior = getattr(module, key)
        setattr(module, key, value)
        stack.callback(setattr, module, key, prior)

    with ExitStack() as stack:
        L.install(stack, dependencies, owner, replace)
        if case == 'initial_owner':
            with pytest.raises(ValueError, match='late_loader_initial_owner'):
                dependencies.session_creator(lambda *args: None, (1, 2))
            return
        selected = dependencies.session_creator(original, (1, 2))
        if case == 'closed':
            stack.close()
        elif case == 'foreign_module':
            monkeypatch.setitem(sys.modules, bootstrap.__name__, N())
        elif case == 'foreign_hook':
            bootstrap.load = lambda *args: None
        elif case in ('same_source_replaced', 'hook_released'):
            def hooked(*args: Any) -> Any:
                return original(*args)
            hooked.__code__ = hooked.__code__.replace(co_filename=str(L.PATCH))
            bootstrap.load = hooked
            assert selected('alias', 'path') == ('alias', 'path', None)
            from types import FunctionType
            bootstrap.load = original if case == 'hook_released' else FunctionType(
                hooked.__code__, hooked.__globals__, closure=hooked.__closure__)
        if case == 'normal':
            assert selected('alias', 'path', {'key': 1}) == ('alias', 'path', {'key': 1})
        else:
            with pytest.raises(ValueError, match='late_loader_'):
                selected('alias', 'path')
    assert dependencies.session_creator is creator
