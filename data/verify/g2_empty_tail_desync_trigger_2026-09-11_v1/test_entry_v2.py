"""計装の汎用署名を許し、実Guard署名を唯一の引数仕様とする輸送対照。"""
from __future__ import annotations
from contextlib import ExitStack
from types import FunctionType, MethodType, SimpleNamespace as N
from typing import Any
import pytest
import entry_boundary_v2 as E
import test_entry_boundary as T


def sample() -> tuple[Any, Any, Any]:
    pipe, state, lease = T.sample()
    controller = object()
    actual = FunctionType(T.Pipe.update.__code__, dict(T.Pipe.update.__globals__, __next_live=controller))
    lease.guard = N(pipe=pipe, actual=actual, controller=controller)
    return pipe, state, lease


@pytest.mark.parametrize('keywords', [False, True])
def test_generic_outer_wrapper_uses_actual_signature(keywords: bool) -> None:
    pipe, state, lease = sample()
    original = pipe.update
    def generic(self: Any, *args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs)
    previous = pipe.update = MethodType(generic, pipe)
    with ExitStack() as stack:
        entry = E.install(stack, pipe, state, lease, object(), T.FRAME)
        if keywords: result = pipe.update(frame_idx=T.FRAME, time_sec=T.FRAME/60, frame=None)
        else: result = pipe.update(T.FRAME, T.FRAME/60, None)
        assert entry.done and pipe.order == ['reset', 'update'] and result is pipe.result
    assert pipe.update is previous


@pytest.mark.parametrize('case', ['pipe', 'controller', 'signature'])
def test_actual_signature_foreign_bindings_rejected(case: str) -> None:
    pipe, state, lease = sample()
    if case == 'pipe': lease.guard.pipe = object()
    if case == 'controller': lease.guard.controller = object()
    if case == 'signature':
        def generic(*args: Any, **kwargs: Any) -> Any: return None
        lease.guard.actual = FunctionType(generic.__code__, dict(generic.__globals__, __next_live=lease.guard.controller))
    with ExitStack() as stack, pytest.raises(ValueError, match='reset_entry:'):
        E.install(stack, pipe, state, lease, object(), T.FRAME)
    assert not pipe.order and 'update' not in vars(pipe)


def test_later_class_probe_is_not_silently_shadowed() -> None:
    pipe, state, lease = sample()
    original = T.Pipe.update
    with ExitStack() as stack:
        entry = E.install(stack, pipe, state, lease, object(), T.FRAME)
        def probe(self: Any, *args: Any, **kwargs: Any) -> Any:
            return original(self, *args, **kwargs)
        T.Pipe.update = probe
        try:
            with pytest.raises(ValueError, match='class_probe_changed_after_install'):
                pipe.update(T.FRAME, T.FRAME/60, None)
            assert entry.error is not None and not pipe.order
        finally:
            T.Pipe.update = original


def test_missing_metadata_has_named_sticky_failure() -> None:
    pipe, state, lease = sample()
    state.clear()
    with ExitStack() as stack:
        entry = E.install(stack, pipe, state, lease, object(), T.FRAME)
        with pytest.raises(ValueError, match='metadata_sink_absent') as first:
            pipe.update(T.FRAME, T.FRAME/60, None)
        assert entry.error is first.value and not pipe.order
