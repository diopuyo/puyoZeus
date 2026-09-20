"""凍結v7の追跡mode生成後へ読み取り専用の局所値採録を追加する。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'g2_exit_next_runtime_2026-09-12_v7'
DIAG = ROOT.parent / 'g2_basis_cascade_native_overlap_2026-09-12_v1'
SPEC = importlib.util.spec_from_file_location('_g2_tracking_observation_base', BASE / 'owned_adapter.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
PRIOR, START, SIDE, protect = A.PRIOR, A.START, A.SIDE, A.protect


def sources() -> tuple[Path, ...]:
    return tuple(sorted(set(A.sources()) | {ROOT / 'owned_adapter.py',
                        BASE / 'owned_adapter.py', DIAG / 'tracking_observation.py'}))


def configured(stack: Any) -> Any:
    previous = A.A.wrap_modules
    captured: dict[Any, Any] = {}
    def wrap_modules(original: Any, inner_stack: Any, owned: dict) -> Any:
        original_modules = previous(original, inner_stack, owned)
        def modules() -> Any:
            parts = original_modules()
            mode = parts.mode
            if mode in captured:
                A.A.C.require(mode.install is captured[mode], 'tracking_install_owner')
                return parts
            native = mode.install
            A.A.C.require(Path(native.__code__.co_filename).resolve() ==
                ROOT.parent / 'g2_basis_cascade_candidate_2026-09-11_v1/mode_v2.py',
                'original_tracking_install')
            owner = sys.modules[A.A.A.OWNED_ALIAS]
            diagnostic = owner.bootstrap().load('_g2_tracking_local_diagnostic',
                                               DIAG / 'tracking_observation.py')
            def install(scope: Any, connection: Any, state: dict) -> Any:
                A.A.C.require(mode.install is install, 'tracking_install_binding')
                value = native(scope, connection, state)
                A.A.C.require(type(value) is mode.Mode and
                              state['probabilistic_tracking_mode'] is value, 'actual_tracking_mode')
                diagnostic.install(scope, value, state['output'], A.A.C.LOCAL.capture)
                return value
            A.A.A.replace_owned(inner_stack, mode, 'install', install)
            captured[mode] = install
            return parts
        return modules
    A.A.A.replace_owned(stack, A.A, 'wrap_modules', wrap_modules)
    return A.configured(stack)
