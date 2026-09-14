"""元Archiveでanchor→登録解除→専用finish検査を連続させる。"""
from __future__ import annotations
import contextlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
import anchor_v2 as A
import test_original_close as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'g2_hidden_basis_initialization_2026-09-11_v1'))
import finish_v2 as FINISH

modules, fixture, actual = F.modules, F.fixture, F.actual


@pytest.mark.parametrize('fault', ['none', 'before_close', 'after_close', 'body_error'])
def test_close_order_with_actual_archive(actual: Any, tmp_path: Path, fault: str) -> None:
    archive = F.F.X.Archive(actual.local.controller, actual.binding)
    factory = N()
    registry = N(factory=factory, current=lambda _: N(frame=34944, scope=('test', '1P')))
    connection = N(registry=registry, binding=N(initial_call_token='step:172'))
    setattr(factory, A.KEY, registry)
    state = dict(output=tmp_path, probabilistic_basis_connection=connection,
                 repeat_scope_guard=N(reset_lease=N(archive=archive)))
    state[A.KEY] = registry
    name, error = type(actual.basis).__module__, ValueError('元body例外')
    original = sys.modules[name]
    def release() -> None:
        sys.modules.pop(name)
        if fault == 'after_close': actual.basis.frame += 2
    try:
        try:
            with contextlib.ExitStack() as stack:
                stack.callback(release)
                A.install(N(factory=factory, state=state, stack=stack))
                if fault == 'before_close': actual.basis.frame += 2
                if fault == 'body_error': raise error
        except ValueError as caught:
            assert fault == 'body_error' and caught is error
        assert name not in sys.modules
        report = json.loads((tmp_path / 'PROBABILISTIC_FACTORY_ANCHOR.json').read_bytes())
        assert report['ready'] is (fault != 'before_close')
        if fault == 'after_close':
            with pytest.raises(AssertionError): FINISH.retained(state, state['repeat_scope_guard'].reset_lease)
        elif fault != 'before_close':
            FINISH.retained(state, state['repeat_scope_guard'].reset_lease)
        if fault == 'body_error': assert report['body_error'] == repr(error)
    finally:
        sys.modules[name] = original
