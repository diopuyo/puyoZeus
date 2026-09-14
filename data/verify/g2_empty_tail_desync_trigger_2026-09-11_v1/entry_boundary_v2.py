"""外側の*args計装でなく、原Guardが認証した実updateの署名に束縛する。"""
from __future__ import annotations
from contextlib import ExitStack
import inspect
from types import MethodType
from typing import Any
import entry_boundary as E


def actual_signature(lease: Any, pipe: Any) -> Any:
    guard = lease.guard
    E.require(guard.pipe is pipe and inspect.isfunction(guard.actual), 'actual_guard')
    E.require(guard.actual.__globals__.get('__next_live') is guard.controller, 'actual_controller')
    signature = inspect.signature(guard.actual, follow_wrapped=False)
    E.require(tuple(signature.parameters) == ('self', 'frame_idx', 'time_sec', 'frame'),
              'original_update_signature')
    return signature


def invoke(entry: E.Entry, original: Any, class_update: Any, signature: Any,
           pipe: Any, args: Any, kwargs: Any) -> Any:
    try:
        E.require(type(pipe).update is class_update, 'class_probe_changed_after_install')
        E.require('collector_metadata_sink' in entry.state, 'metadata_sink_absent')
        values = signature.bind(pipe, *args, **kwargs).arguments
        entry.before(pipe, values['frame_idx'], values['time_sec'])
        return original(*args, **kwargs)
    except BaseException as exc:
        entry.error = entry.error or exc
        raise entry.error


def install(stack: ExitStack, pipe: Any, state: Any, lease: Any,
            recovery: Any, frame: int) -> E.Entry:
    original = pipe.update
    E.require(inspect.ismethod(original) and original.__self__ is pipe, 'actual_update_method')
    signature = actual_signature(lease, pipe)
    entry = E.Entry(pipe, state, lambda f,t: lease.perform(recovery,f,t), frame)
    class_update = type(pipe).update
    existed, previous = 'update' in vars(pipe), vars(pipe).get('update')
    def update(self: Any, *args: Any, **kwargs: Any) -> Any:
        return invoke(entry, original, class_update, signature, self, args, kwargs)
    def restore() -> None:
        if existed: pipe.update = previous
        else: vars(pipe).pop('update', None)
    stack.callback(restore)
    pipe.update = MethodType(update, pipe)
    return entry
