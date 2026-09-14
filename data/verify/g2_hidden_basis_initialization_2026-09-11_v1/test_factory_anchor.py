"""close前の別保存・所有参照の差替拒否・元例外保持を人工contextで検査する。"""
from __future__ import annotations
import contextlib
import json
from types import SimpleNamespace as N
from typing import Any
import pytest
import factory_anchor as A


def objects(output: Any, stack: Any) -> Any:
    factory = N()
    registry = N(factory=factory, current=lambda _: N(frame=34944, scope=('s', 'r', 1, 2, 3, 1, '1P')))
    connection = N(registry=registry, binding=N(initial_call_token='step:172'))
    setattr(factory, A.KEY, registry)
    state = dict(output=output, probabilistic_basis_connection=connection)
    state[A.KEY] = registry
    return N(factory=factory, state=state, stack=stack)


@pytest.mark.parametrize('fault', ['none', 'factory_swap', 'connection_swap'])
def test_before_original_close_anchor(tmp_path: Any, fault: str) -> None:
    with contextlib.ExitStack() as stack:
        context = objects(tmp_path, stack)
        stack.callback(delattr, context.factory, A.KEY)
        A.install(context)
        if fault == 'factory_swap': setattr(context.factory, A.KEY, object())
        if fault == 'connection_swap': context.state['probabilistic_basis_connection'].registry = object()
    report = json.loads((tmp_path / 'PROBABILISTIC_FACTORY_ANCHOR.json').read_bytes())
    assert not hasattr(context.factory, A.KEY)
    assert report['ready'] is (fault == 'none')
    if fault == 'none':
        assert report['registry_id'] == id(context.state[A.KEY]) and report['source_call_token'] == 'step:172'
    else:
        assert report['error'] is not None


def test_anchor_does_not_hide_body_error(tmp_path: Any) -> None:
    error = ValueError('元body例外')
    with pytest.raises(ValueError) as caught:
        with contextlib.ExitStack() as stack:
            context = objects(tmp_path, stack)
            A.install(context)
            raise error
    assert caught.value is error
    report = json.loads((tmp_path / 'PROBABILISTIC_FACTORY_ANCHOR.json').read_bytes())
    assert report['body_error'] == repr(error)
