"""実Scheduledに対するloader再入を、元のscope所有で検査する。"""
from contextlib import ExitStack
from types import SimpleNamespace as N
from typing import Any
import pytest
from test_compatibility import module, replace
import compatibility as C
import selection as S


@pytest.mark.parametrize('case', ['same', 'changed', 'closed'])
def test_loader_scope_reentry(module: Any, case: str) -> None:
    original = module.Session.completed
    bootstrap = N(load=lambda alias, path, injection=None: module)
    owner = N(bootstrap=lambda: bootstrap)
    inner, other = ExitStack(), ExitStack()
    selected = [inner]
    try:
        with ExitStack() as outer:
            S.install_owner(outer, owner, replace, lambda: selected[0])
            load = owner.bootstrap().load
            assert load('first', S.TARGET) is module
            bound = module.Session.completed
            assert bound is not original
            if case == 'changed': selected[0] = other
            if case == 'closed': inner.close()
            if case == 'same':
                assert load('again', S.TARGET) is module
                assert module.Session.completed is bound
            else:
                with pytest.raises(ValueError, match='legacy_m1_compatibility:scope_'):
                    load('again', S.TARGET)
    finally:
        inner.close()
        other.close()
    assert module.Session.completed is original
