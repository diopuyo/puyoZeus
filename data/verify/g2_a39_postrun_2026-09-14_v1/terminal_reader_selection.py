"""原moduleの来歴/alias所有を保持し、最終reader関数だけを同じ寿命へ接続する。"""
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import sys
import terminal_prefix_reader as R

TARGET = Path(__file__).resolve().parent.parent / 'g2_async_projected_evaluation_2026-09-13_v1/prefix_live_reader.py'
TRACE: list[dict] = []


def install_owner(stack: Any, owner: Any, replace: Any, scope_getter: Any) -> None:
    previous, installed = owner.bootstrap, []
    def bootstrap() -> Any:
        value = previous()
        if installed:
            if installed[0] is not value: raise ValueError('terminal_reader_bootstrap_changed')
            return value
        if any(getattr(m, '__file__', None) and Path(m.__file__).resolve() == TARGET
               for m in tuple(sys.modules.values())):
            raise ValueError('terminal_reader_late_install')
        original, bound = value.load, {}
        def load(alias: str, path: Any, injection: Any = None) -> Any:
            module = original(alias, path, injection)
            if Path(path).resolve() != TARGET: return module
            if alias in bound:
                old, lifetime = bound[alias]
                if old is not module or lifetime.closed: raise ValueError('terminal_reader_owner_changed')
                return module
            if Path(module.__file__).resolve() != TARGET: raise ValueError('terminal_reader_source')
            scope, lifetime = scope_getter(), N(closed=False)
            scope.callback(setattr, lifetime, 'closed', True)
            replace(scope, module, 'read', R.read)
            bound[alias] = (module, lifetime)
            TRACE.append(dict(alias=alias, target=str(TARGET), selected=R.read.__code__.co_filename))
            return module
        replace(stack, value, 'load', load)
        installed.append(value)
        return value
    replace(stack, owner, 'bootstrap', bootstrap)
