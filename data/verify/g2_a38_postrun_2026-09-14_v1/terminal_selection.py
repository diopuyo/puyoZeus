"""元loaderの生成直後・Session構築前にだけ終了候補を接続する。実走未採用。"""
from __future__ import annotations
from pathlib import Path
from types import FunctionType, SimpleNamespace as N
from typing import Any
import sys
import first_terminal_candidate as FIRST
import second_inactive_candidate as SECOND
import first_terminal_saved as SAVED
import terminal_publication_boundary as BOUNDARY
import terminal_boundary as C

VERIFY = Path(__file__).resolve().parent.parent
TRACE: list[dict] = []  # 実bind到達の診断票。終了品質や公開権限には使わない。
ARRIVAL = VERIFY/'g2_arrival_ack_candidate_2026-09-12_v1'
TARGETS = {
    ARRIVAL/'arrival_mode.py': 'first',
    ARRIVAL/'arrival_final_saved.py': 'saved',
    VERIFY/'g2_second_terminal_arrival_2026-09-14_v1/session_connection.py': 'second',
    VERIFY/'g2_live_probability_context_2026-09-12_v1/probability_finish.py': 'boundary',
}


def bind(module: Any, kind: str, stack: Any, replace: Any) -> None:
    if kind == 'first':
        mode, capture = FIRST.derive(module)
        replace(stack, module, 'Mode', mode)
        replace(stack, module.E, 'Capture', capture)
    elif kind == 'second':
        replace(stack, module, 'mode_class', SECOND.mode_factory(module.mode_class))
    elif kind == 'saved':
        prefix = sys.modules.get('_g2_prefix_final')
        C.require(prefix is not None and Path(prefix.__file__).resolve() ==
            VERIFY/'g2_prefix_lane_integration_2026-09-13_v1/prefix_final_saved.py', 'terminal_actual_prefix_saved')
        previous = N(**vars(prefix))
        def verify(original: Any, mode: Any, state: dict, serializer: Any, native: Any) -> dict:
            return SAVED.verify_prefix(previous, original, mode, state, serializer, native)
        replace(stack, prefix, 'verify', verify)  # 元prefix_selectionが保持するmodule.verifyの参照を維持。
    elif kind == 'boundary':
        previous, original = module.prepared, module.BOUNDARY
        def prepared(owner: Any, factory: Any, state: dict, lease: Any) -> dict:
            def check(*args: Any, **kwargs: Any) -> dict:
                return BOUNDARY.check(original, state['probabilistic_tracking_mode'], *args, **kwargs)
            function = FunctionType(previous.__code__, dict(previous.__globals__, BOUNDARY=N(check=check)),
                previous.__name__, previous.__defaults__, previous.__closure__)
            return function(owner, factory, state, lease)
        replace(stack, module, 'prepared', prepared)
    else:
        C.require(False, 'terminal_selection_kind')
    TRACE.append(dict(kind=kind, target=str(Path(module.__file__).resolve())))


def wrap_load(original: Any, scope_getter: Any, replace: Any, prebound: dict | None = None) -> Any:
    bound = dict(prebound or {})
    records = [{'target':str(path), 'kind':TARGETS[path], 'prebound':True} for path in bound]
    def load(alias: str, path: Any, injection: Any = None) -> Any:
        target = Path(path).resolve()
        value = original(alias, path, injection)
        if target not in TARGETS:
            return value
        if target in bound:
            owned, lifetime = bound[target]
            C.require(owned is value and not lifetime.closed, 'terminal_selection_module_replaced_or_closed')
            return value
        scope, lifetime = scope_getter(), N(closed=False)
        scope.callback(setattr, lifetime, 'closed', True)
        C.require(Path(value.__file__).resolve() == target, 'terminal_selection_source')
        bind(value, TARGETS[target], scope, replace)
        bound[target] = (value, lifetime)
        records.append(dict(target=str(target), kind=TARGETS[target], prebound=False))
        return value
    load.terminal_bound_records = records
    return load


def install_owner(stack: Any, owner: Any, replace: Any, scope_getter: Any) -> None:
    previous, installed = owner.bootstrap, []
    def bootstrap() -> Any:
        value = previous()
        if installed:
            C.require(installed[0] is value, 'terminal_selection_bootstrap')
            return value
        existing = {}
        for module in tuple(sys.modules.values()):
            path = getattr(module, '__file__', None)
            target = None if path is None else Path(path).resolve()
            if target not in TARGETS:
                continue
            C.require(TARGETS[target] == 'boundary' and target not in existing,
                'terminal_selection_late_install:' + str(target))
            bind(module, 'boundary', stack, replace)
            lifetime = N(closed=False)
            stack.callback(setattr, lifetime, 'closed', True)
            existing[target] = (module, lifetime)
        replace(stack, value, 'load', wrap_load(value.load, scope_getter, replace, existing))
        installed.append(value)
        return value
    replace(stack, owner, 'bootstrap', bootstrap)
