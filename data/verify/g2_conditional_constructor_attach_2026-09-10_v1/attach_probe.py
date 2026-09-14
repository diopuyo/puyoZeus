"""原 attach と原4更新を囲み、追加参照の復元だけを計測する。"""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import inspect
import sys
from types import FunctionType, SimpleNamespace as N
from typing import Any

SCOPE_NAMES = ('pipe.reset', 'type(pipe).update', 'N.begin', 'N.enqueue', 'N.end', 'N.reset')


def facade(base: Any, scope: Any) -> Any:
    def references(factory: Any, pipe: Any, state: Any) -> tuple[Any, ...]:
        return base.references(factory, pipe, state) + scope.references(factory, pipe)
    value = N(**(vars(base) | dict(references=references)))
    assert value.install is base.install and value.P is base.P
    return value


def reference(value: Any) -> dict[str, Any]:
    function = getattr(value, '__func__', value)
    code = getattr(function, '__code__', None)
    return dict(function_id=id(function), self_id=id(getattr(value, '__self__', None)),
        name=getattr(function, '__qualname__', type(function).__name__),
        source=None if code is None else code.co_filename,
        code_sha256=None if code is None else hashlib.sha256(code.co_code).hexdigest())


def profile(targets: dict[str, Any], previous: Any, frames: dict[str, list[Any]]) -> Any:
    def observe(frame: Any, event: str, result: Any) -> None:
        if previous is not None:
            previous(frame, event, result)
        if event != 'call':
            return
        for name, code in targets.items():
            if frame.f_code is code and all(frame is not old for old in frames[name]):
                frames[name].append(frame)
    return observe


def accepted(evidence: dict[str, Any], state: Any, factory: Any) -> None:
    assert evidence['calls'] == dict(attach=1, install=1, v5=1, update=4)
    assert evidence['first_update_after_attach'] and evidence['closed']
    assert evidence['references_restored'] and all(evidence['reference_equal'])
    assert all(evidence['scope_reference_equal'].values())
    assert evidence['qualification']['qualified_restored'] and evidence['profile_restored']
    assert state['conditional_full_installs'] == 1
    assert state['combined_install']['configure_calls'] == 1
    assert state['conditional_revision_connection']['installs'] == 1
    guard = state['repeat_scope_guard']
    assert guard.error is None and guard.record is None and guard.frame == 29058
    assert guard.binding is None and not factory.controller.history
    assert not state['conditional_full_rows']
    evidence.update(scope_guard_last_frame=guard.frame, baseline_owned=False,
        configure_calls=1, revision_installs=1, scope_guard_reinstalled=False)


def run(original: Any, hook: Any, candidate: Any, scope: Any, output: Any,
        m: Any, state: Any, real: Any, fixture: Any, binding: Any, factory: Any) -> Any:
    pipe = real[0]
    assert not factory.controller.history
    before = candidate.references(factory, pipe, state)
    scope_before = scope.references(factory, pipe)
    previous = sys.getprofile()
    targets = dict(attach=hook.attach.__code__, install=candidate.install.__code__,
        v5=inspect.unwrap(candidate.P.Q.R.Q.installed).__code__, update=type(pipe).update.__code__)
    frames: dict[str, list[Any]] = {name: [] for name in targets}
    evidence: dict[str, Any] = dict(installed=False, closed=False, qualification={}, rows=[],
        live_constructor_verified=False, references_before=[reference(v) for v in before])
    try:
        with ExitStack() as stack:
            stack.callback(sys.setprofile, previous)
            sys.setprofile(profile(targets, previous, frames))
            hook.attach(stack, candidate, factory, pipe, state, evidence)
            evidence['first_update_after_attach'] = not frames['update']
            assert evidence['first_update_after_attach']
            result = original(m, state, real, fixture, binding, factory)
    finally:
        after = candidate.references(factory, pipe, state)
        evidence.update(calls={name: len(values) for name, values in frames.items()},
            reference_equal=[a == b for a, b in zip(before, after, strict=True)],
            references_after=[reference(v) for v in after],
            scope_reference_equal=dict(zip(SCOPE_NAMES,
                (a == b for a, b in zip(scope_before, scope.references(factory, pipe), strict=True)))),
            profile_restored=sys.getprofile() is previous)
        candidate.K.write(output/'ATTACH.json', evidence)
    accepted(evidence, state, factory)
    candidate.K.write(output/'ATTACH_ACCEPTED.json', evidence)
    return result


def derived(smoke: Any, hook: Any, candidate: Any, scope: Any, output: Any) -> Any:
    def drive(*args: Any) -> Any:
        return run(smoke.drive, hook, candidate, scope, output, *args)
    original = smoke.execute
    value = FunctionType(original.__code__, dict(original.__globals__, drive=drive),
        original.__name__, original.__defaults__, original.__closure__)
    assert value.__code__ is original.__code__ and value.__closure__ is original.__closure__
    assert all(value.__globals__[key] is item for key, item in original.__globals__.items() if key != 'drive')
    return value
