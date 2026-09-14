"""contextmanagerの包装ではなく検証対象の原本体globalsを束縛する。"""
from __future__ import annotations
from types import FunctionType, SimpleNamespace
from typing import Any
import settled_connection as C


def install(stack: Any, control: Any, state: Any, patch: Any, revision: Any,
            rows: list[Any], next_module: Any) -> None:
    cls, old = type(control), type(control).hold_transition
    body = old.__wrapped__
    value = SimpleNamespace(**body.__globals__)
    assert body.__name__ == 'hold_transition' and value.hold_transition is old
    def hold(self: Any, sm: Any, signals: Any, binding: Any) -> Any:
        return C.H.installed(self, sm, signals, binding, value)
    patch(stack, cls, 'hold_transition', hold)
    C.install_policy(stack, control, patch, revision, rows)
    C.U.install(stack, state, patch)
    def ready(ticket: Any, view: Any) -> bool:
        return getattr(ticket, 'current_published', False) and next_module.ready(ticket, view)
    install_next = FunctionType(next_module.install.__code__, dict(vars(next_module), ready=ready))
    install_next(stack, control, patch, rows)
