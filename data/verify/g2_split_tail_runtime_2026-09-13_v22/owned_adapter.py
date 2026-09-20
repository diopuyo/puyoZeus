"""元v21を保持し、終局までの有限観測範囲だけを明示選択する。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
from typing import Any
import window as W

ROOT = Path(__file__).resolve().parent
PREVIOUS_ROOT = ROOT.parent / 'g2_m1_second_runtime_2026-09-13_v21'
SPEC = importlib.util.spec_from_file_location('_split_previous_v21', PREVIOUS_ROOT / 'owned_adapter.py')
PREVIOUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREVIOUS)
A = PREVIOUS.A
PRIOR, START, SIDE, protect = PREVIOUS.PRIOR, PREVIOUS.START, PREVIOUS.SIDE, PREVIOUS.protect
CANDIDATE, SAFETY, BASE = PREVIOUS.CANDIDATE, PREVIOUS.SAFETY, PREVIOUS.BASE
UNBOUND_MODULE, NOTICE = PREVIOUS.UNBOUND_MODULE, PREVIOUS.NOTICE
PREVIOUS_ADAPTER = PREVIOUS.PREVIOUS_ADAPTER


def sources() -> tuple[Path, ...]:
    return tuple(sorted(set(PREVIOUS.sources()) | {ROOT / 'owned_adapter.py', ROOT / 'window.py'}))


def configured(stack: Any) -> Any:
    selected = PREVIOUS.configured(stack)
    replace = A.A.A.V4.replace_owned
    namespace = selected.__globals__
    modules = (namespace['K'], namespace['S'].K, namespace['Q'].K)
    for module in {id(value): value for value in modules}.values():
        W.common(stack, module, replace)
    adapter = sys.modules['live_adapter']
    if Path(adapter.__file__).resolve() != PRIOR / 'live_adapter.py' or adapter.END_FRAME != W.OLD_LAST:
        raise ValueError('split_window_live_adapter')
    replace(stack, adapter, 'END_FRAME', W.LAST)
    owner = sys.modules[A.A.A.V4.OWNED_ALIAS]
    original, installed = owner.bootstrap, []
    def bootstrap() -> Any:
        value = original()
        if not installed:
            W.load_wrapper(stack, value, replace)
            installed.append(value)
        elif installed[0] is not value:
            raise ValueError('split_window_bootstrap_owner')
        return value
    replace(stack, owner, 'bootstrap', bootstrap)
    return selected
