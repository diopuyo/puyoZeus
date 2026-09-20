"""v2の全終了処理を維持し、原capture観測を共有設置に一回追加する。"""
from __future__ import annotations
from pathlib import Path
from types import FunctionType, SimpleNamespace as N
import importlib.util
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'g2_conditional_live_adapter_2026-09-10_v2'
OBS = ROOT.parent / 'g2_conditional_current_call_evidence_2026-09-10_v2'
JOIN = ROOT.parent / 'g2_conditional_current_call_join_2026-09-10_v1'


def load(alias: str, path: Path, stack: Any) -> Any:
    assert alias not in sys.modules
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    stack.callback(sys.modules.pop, alias, None)
    spec.loader.exec_module(module)
    return module


def connect(stack: Any, previous: Any, main: Any, observer: Any, connection: Any) -> None:
    repeated, common = main.__globals__['REPEAT'], main.__globals__['Q'].K
    supplied, checked = repeated.load, repeated.verify
    def replace(inner: Any, obj: Any, name: str, value: Any) -> None:
        old = getattr(obj, name)
        inner.callback(setattr, obj, name, old)
        setattr(obj, name, value)
    def load_candidate(inner: Any) -> Any:
        candidate, hook = supplied(inner)
        def installed(scope: Any, factory: Any, pipe: Any, state: Any, rows: Any) -> None:
            candidate.install(scope, factory, pipe, state, rows)
            connection.install(scope, observer, factory, state, replace, common.write)
        return N(**(vars(candidate) | dict(install=installed))), hook
    def verify(state: Any) -> Any:
        result = checked(state)
        connection.verify(state)
        return result | dict(original_capture_observer_closed=True)
    replace(stack, repeated, 'load', load_candidate)
    replace(stack, repeated, 'verify', verify)


def configured(stack: Any) -> Any:
    previous = load('_conditional_v2_adapter', PREVIOUS / 'adapter.py', stack)
    main = previous.configured(stack)
    paths = list(sys.path)
    stack.callback(setattr, sys, 'path', paths)
    sys.path.insert(0, str(OBS))
    observer = load('_conditional_capture_observer', OBS / 'observer.py', stack)
    connection = load('_conditional_capture_connection', ROOT / 'call_connection.py', stack)
    connect(stack, previous, main, observer, connection)
    common = main.__globals__['Q'].K
    original = common.guards
    pins = {str(path): common.sha(path) for directory in (ROOT, OBS, JOIN)
        for path in (*directory.glob('*.py'), directory / 'PLAN.md')}
    def guards() -> Any:
        assert all(common.sha(Path(path)) == digest for path, digest in pins.items()), 'capture_source_changed'
        return original() | pins
    stack.callback(setattr, common, 'guards', original)
    common.guards = guards
    local = N(**(vars(main.__globals__['K']) | dict(ROOT=ROOT, guards=guards)))
    return FunctionType(main.__code__, dict(main.__globals__, K=local),
        main.__name__, main.__defaults__, main.__closure__)
