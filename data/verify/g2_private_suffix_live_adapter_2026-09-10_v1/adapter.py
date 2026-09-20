"""原v3 configuredとfinish本体を保持し、新設置/最終集計口だけ追加する。"""
from __future__ import annotations
from pathlib import Path
from types import FunctionType, SimpleNamespace as N
import importlib.util
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'g2_conditional_live_adapter_2026-09-10_v3'
FINAL_KEY = 'private_suffix_original_finalizer'


def load(alias: str, path: Path, stack: Any) -> Any:
    assert alias not in sys.modules
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    stack.callback(sys.modules.pop, alias, None)
    spec.loader.exec_module(module)
    return module


def patch(stack: Any, obj: Any, name: str, value: Any) -> None:
    stack.callback(setattr, obj, name, getattr(obj, name))
    setattr(obj, name, value)


def connect(stack: Any, main: Any, extra: Any, modules: Any, finalizer: Any) -> None:
    repeated, closure = main.__globals__['REPEAT'], main.__globals__['Q']
    supplied, original = repeated.load, closure.FINAL.evaluate
    def load_candidate(inner: Any) -> Any:
        candidate, hook = supplied(inner)
        def install(scope: Any, factory: Any, pipe: Any, state: Any, rows: Any) -> None:
            assert FINAL_KEY not in state and extra.FACTORY not in state, 'private_suffix_duplicate_install'
            candidate.install(scope, factory, pipe, state, rows)
            state[FINAL_KEY] = original
            extra.install(scope, factory, pipe, state, modules, closure.K.write)
        return N(**(vars(candidate) | dict(install=install))), hook
    def evaluate(goals: Any, rows: Any, legal: Any, output: Any, state: Any) -> Any:
        assert state[FINAL_KEY] is original, 'private_suffix_finalizer_origin'
        return finalizer.evaluate(goals, rows, legal, output, state)
    patch(stack, repeated, 'load', load_candidate)
    patch(stack, closure.FINAL, 'evaluate', evaluate)


def guards(stack: Any, common: Any, extra: Any) -> Any:
    previous = common.guards
    folders = (ROOT, extra.SUFFIX, extra.PRIVATE, extra.EVIDENCE,
        ROOT.parent/'g2_private_suffix_finalizer_stage1_2026-09-10_v1',
        ROOT.parent/'g2_private_suffix_world_2026-09-10_v1',
        ROOT.parent/'g2_private_suffix_fusion_2026-09-10_v1')
    pins = {str(path): common.sha(path) for folder in folders
        for path in folder.glob('*.py')}
    pins[str(ROOT/'PLAN.md')] = common.sha(ROOT/'PLAN.md')
    def checked() -> dict[str, str]:
        assert all(common.sha(Path(path)) == digest for path, digest in pins.items()), 'private_suffix_source_changed'
        return previous() | pins
    patch(stack, common, 'guards', checked)
    return checked


def configured(stack: Any) -> Any:
    original_path, values = sys.path, list(sys.path)
    def restore() -> None:
        original_path[:] = values
        sys.path = original_path
    stack.callback(restore)
    previous = load('_private_live_previous', PREVIOUS/'adapter.py', stack)
    main = previous.configured(stack)
    extra = load('_private_live_extra', ROOT/'extra.py', stack)
    references = load('_private_live_references', ROOT/'references.py', stack)
    modules = extra.load_modules(stack, load, references)
    finalizer = load('_private_live_finalizer', ROOT/'live_finalizer.py', stack)
    closure, before = main.__globals__['Q'], main.__globals__['Q'].finish
    connect(stack, main, extra, modules, finalizer)
    checked = guards(stack, closure.K, extra)
    local = N(**(vars(main.__globals__['K']) | dict(ROOT=ROOT, guards=checked)))
    result = FunctionType(main.__code__, dict(main.__globals__, K=local),
        main.__name__, main.__defaults__, main.__closure__)
    assert closure.finish is before and result.__code__ is main.__code__
    return result
