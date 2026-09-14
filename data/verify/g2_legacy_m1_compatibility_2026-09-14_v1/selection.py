"""元Session合成前に互換観測を接続し、所有scopeで元methodへ戻す。"""
from __future__ import annotations
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import compatibility as C

ROOT = Path(__file__).resolve().parent
TARGET = ROOT.parent / 'g2_m1_completion_candidate_2026-09-12_v1/scheduled_session.py'


def bind(module: Any, stack: Any, replace: Any) -> None:
    before_completed, before_close = module.Session.completed, module.Session.close
    C.require(Path(before_completed.__code__.co_filename).resolve() == TARGET,
              'scheduled_completed_source')
    def completed(self: Any, frame: int) -> Any:
        return C.completed(before_completed, self, frame)
    def close(self: Any, kind: Any, body: Any, trace: Any) -> bool:
        return C.closed(before_close, self, kind, body, trace)
    replace(stack, module.Session, 'completed', completed)
    replace(stack, module.Session, 'close', close)


def install_owner(stack: Any, owner: Any, replace: Any, scope_getter: Any) -> None:
    original_bootstrap, installed, bound = owner.bootstrap, [], []
    lifetime = N(scope=None, closed=False)
    def bootstrap() -> Any:
        value = original_bootstrap()
        if installed:
            C.require(installed[0] is value, 'bootstrap_owner')
            return value
        original_load = value.load
        def load(alias: str, path: Any, injection: Any = None) -> Any:
            if Path(path).resolve() == TARGET:
                scope = scope_getter()
                C.require(not lifetime.closed, 'scope_closed')
                C.require(lifetime.scope is None or lifetime.scope is scope, 'scope_changed')
                if lifetime.scope is None:
                    lifetime.scope = scope
                    scope.callback(setattr, lifetime, 'closed', True)
            module = original_load(alias, path, injection)
            if Path(path).resolve() == TARGET and module not in bound:
                bind(module, lifetime.scope, replace)
                bound.append(module)
            return module
        C.require(not any(getattr(m, '__file__', None) and
            Path(m.__file__).resolve() == TARGET for m in tuple(sys.modules.values())),
            'scheduled_loaded_before_install')
        replace(stack, value, 'load', load)
        installed.append(value)
        return value
    replace(stack, owner, 'bootstrap', bootstrap)
