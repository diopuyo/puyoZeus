"""実factory生成後だけ新Contextを装着する。確率finalize未接続のため実動画は禁止。"""
from __future__ import annotations
import importlib.util
import importlib
import inspect
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace as N
from typing import Any

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'g2_empty_tail_reset_live_adapter_2026-09-11_v1'
END_FRAME = 36298  # 対象の元実video38診断区間終端。別動画への一般化ではない。
FINISH_MODULES = ('probability_owner', 'probability_boundary', 'probability_world',
                  'probability_saved', 'probability_stage', 'probability_finish', 'stage_retired_identity')


def install_finish(stack: Any, main: Any) -> None:
    original = inspect.getclosurevars(main.__globals__['Q'].FINAL.evaluate).nonlocals['FINISH']
    require_path = PRIOR / 'live_finish_v2.py'
    assert Path(original.__file__).resolve() == require_path, 'probability_original_finalizer'
    assert not any(name in sys.modules for name in FINISH_MODULES), 'probability_finish_foreign_alias'
    before = list(sys.path)
    try:
        sys.path.insert(0, str(ROOT))
        module = importlib.import_module('probability_finish')
    finally:
        sys.path[:] = before
    owned = {name: sys.modules[name] for name in FINISH_MODULES}
    assert all(Path(value.__file__).resolve() == ROOT / (name + '.py') for name, value in owned.items())
    def restore() -> None:
        assert all(sys.modules.get(name) is value for name, value in owned.items()), 'probability_finish_alias_restore'
        for name in owned:
            sys.modules.pop(name)
    stack.callback(restore)
    module.install(stack, main, original, lambda state: sys.modules['_private_live_finalizer'].context(state))


def load(alias: str, path: Path) -> Any:
    if alias in sys.modules:
        module = sys.modules[alias]
        if Path(module.__file__).resolve() != path:
            raise RuntimeError('live_probability_adapter_alias')
        return module
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def install_outer(stack: Any, main: Any) -> None:
    alias = '_g2_live_probability_outer'
    assert alias not in sys.modules, 'probability_outer_foreign_alias'
    module = load(alias, ROOT / 'probability_outer_scope.py')
    def restore() -> None:
        assert sys.modules.get(alias) is module, 'probability_outer_alias_restore'
        sys.modules.pop(alias)
    stack.callback(restore)
    module.install(stack, main)


def configured(stack: Any) -> Any:
    live = any(arg == '--mode=live' or (arg == '--mode' and sys.argv[i + 1:i + 2] == ['live'])
               for i, arg in enumerate(sys.argv))
    if live:
        raise RuntimeError('probabilistic_live_finalize_not_connected')
    before = list(sys.path)
    stack.callback(sys.path.__setitem__, slice(None), before)
    sys.path.insert(0, str(PRIOR))
    selected = load('_g2_live_probability_original_adapter', PRIOR / 'adapter_v5.py')
    v4, base = selected.A, selected.A.A.A
    factory = load('_g2_live_probability_owned_loader', ROOT / 'loader.py')
    def attach(scope: Any, original_factory: Any, pipe: Any, state: Any, modules: Any) -> None:
        parent = modules.proof.Context
        cls = factory.build(parent, end_frame=END_FRAME)
        proof = N(**(vars(modules.proof) | dict(Context=cls)))
        supplied = N(**(vars(modules) | dict(proof=proof)))
        base.attach(scope, original_factory, pipe, state, supplied)
        state['live_probability_context_class'] = cls
    constructor = FunctionType(base.constructor.__code__, dict(vars(base), attach=attach))
    configure = FunctionType(base.configured.__code__,
                             dict(vars(base), constructor=constructor, dependencies=v4.dependencies))
    main = configure(stack)
    # 非対象/未発火の旧終了検査は残す。確率分岐を整数合格へ昇格させない。
    selected.configured.__globals__['finalize'](stack, main)
    install_finish(stack, main)  # 実動画GOは独立検収・短い統合が閉じるまで引き続き拒否。
    install_outer(stack, main)
    return main
