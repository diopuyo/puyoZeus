"""凍結v1の入口を再用し、原終了処理の最終集計一箇所だけ接続する。"""
from __future__ import annotations
from pathlib import Path
from types import FunctionType, SimpleNamespace as N
import importlib.util
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'g2_conditional_live_adapter_2026-09-10_v1'
FUSION = ROOT.parent / 'g2_conditional_finalizer_fusion_2026-09-10_v1'
WORLD = ROOT.parent / 'g2_conditional_finalizer_world_2026-09-10_v1'
STAGE1 = ROOT.parent / 'g2_conditional_finalizer_compatibility_2026-09-10_v1'


def load(alias: str, path: Path, stack: Any) -> Any:
    assert alias not in sys.modules
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    stack.callback(sys.modules.pop, alias, None)
    spec.loader.exec_module(module)
    return module


def configured(stack: Any) -> Any:
    original_path, original_values = sys.path, list(sys.path)
    def restore_path() -> None:
        original_path[:] = original_values
        sys.path = original_path
    # cv2はimport時にsys.path自体を再代入する。最外側で参照と内容を戻す。
    stack.callback(restore_path)
    previous = load('_conditional_previous_adapter', PREVIOUS / 'adapter.py', stack)
    main = previous.configured(stack)
    paths = list(sys.path)
    stack.callback(sys.path.__setitem__, slice(None), paths)
    sys.path.insert(0, str(FUSION))
    finalizer = load('_conditional_live_finalizer', FUSION / 'live_connection.py', stack)
    closure, common = main.__globals__['Q'], main.__globals__['Q'].K
    original_finish = closure.finish
    previous.patch(stack, closure.FINAL, 'evaluate', finalizer.evaluate)
    original_guards = common.guards
    pins = {str(path): common.sha(path) for directory in (ROOT, FUSION, WORLD, STAGE1)
        for path in (*directory.glob('*.py'), directory / 'PLAN.md')}
    def guards() -> dict[str, str]:
        assert all(common.sha(Path(path)) == digest for path, digest in pins.items()), 'finalizer_source_changed'
        return original_guards() | pins
    previous.patch(stack, common, 'guards', guards)
    local = N(**(vars(main.__globals__['K']) | dict(ROOT=ROOT, guards=guards)))
    result = FunctionType(main.__code__, dict(main.__globals__, K=local),
        main.__name__, main.__defaults__, main.__closure__)
    assert closure.finish is original_finish and result.__code__ is main.__code__
    return result
