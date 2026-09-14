"""凍結v9を保持し、実Contextの到来分離modeと保存検査だけを私有選択する。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'g2_ui_hsv_runtime_2026-09-12_v9'
CANDIDATE = ROOT.parent / 'g2_arrival_ack_candidate_2026-09-12_v1'
SPEC = importlib.util.spec_from_file_location('_g2_arrival_runtime_base', BASE / 'owned_adapter.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
PRIOR, START, SIDE, protect = A.PRIOR, A.START, A.SIDE, A.protect
FILES = ('ledger.py', 'cascade_arrival.py', 'enqueue_capture.py', 'source_replay.py',
         'receipt_replay.py', 'stable_capture.py', 'arrival_mode.py', 'arrival_saved.py',
         'arrival_final_saved.py', 'arrival_loader.py')


def sources() -> tuple[Path, ...]:
    return tuple(sorted(set(A.sources()) | {ROOT / 'owned_adapter.py', BASE / 'owned_adapter.py'}
                        | {CANDIDATE / name for name in FILES}))


def choose(stack: Any, old: Any, captured: dict) -> Any:
    owner = sys.modules[A.V4.OWNED_ALIAS]
    A.V4.require_owned(owner)
    load = owner.bootstrap().load
    loader = load('_g2_arrival_loader', CANDIDATE / 'arrival_loader.py')
    parts = loader.build(old, load)
    final = sys.modules['probability_finish']
    if parts.mode in captured:
        A.V5.C.require(parts.mode.install is captured[parts.mode] and final.SAVED is parts.arrival_saved,
                       'arrival_runtime_selection_changed')
        return parts
    A.V5.C.require(Path(final.SAVED.__file__).resolve() == PRIOR / 'probability_saved.py', 'arrival_original_saved')
    A.V4.replace_owned(stack, final, 'SAVED', parts.arrival_saved)
    original = parts.mode.install
    diagnostic = sys.modules['_g2_tracking_local_diagnostic']
    def install(scope: Any, connection: Any, state: dict) -> Any:
        A.V5.C.require(parts.mode.install is install, 'arrival_install_owner')
        value = original(scope, connection, state)
        A.V5.C.require(type(value) is parts.mode.Mode and state['probabilistic_tracking_mode'] is value,
                       'arrival_actual_mode')
        diagnostic.install(scope, value, state['output'], A.V5.C.LOCAL.capture)
        return value
    A.V4.replace_owned(stack, parts.mode, 'install', install)
    captured[parts.mode] = install
    return parts


def configured(stack: Any) -> Any:
    selected = A.configured(stack)
    previous, captured = A.V5.wrap_modules, {}
    def wrap_modules(original: Any, inner: Any, owned: dict) -> Any:
        original_modules = previous(original, inner, owned)
        def modules() -> Any:
            return choose(inner, original_modules(), captured)
        return modules
    A.V4.replace_owned(stack, A.V5, 'wrap_modules', wrap_modules)
    return selected
