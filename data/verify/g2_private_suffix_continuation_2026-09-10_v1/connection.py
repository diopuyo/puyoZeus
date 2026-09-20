"""残単headの開始と私有配置を接続。既存conditional anchor分岐には割り込まない。"""
from __future__ import annotations
from typing import Any


def tail_capture(stack: Any, history: Any, patch: Any, basis: Any) -> None:
    previous = history.consumed
    def consumed(control: Any, call: Any, caller: Any, rows: list[Any]) -> None:
        previous(control, call, caller, rows)
        if len(call['view'].refs) > 1:
            basis.capture(control, call['binding'], call)
    patch(stack, history, 'consumed', consumed)


def install(stack: Any, factory: Any, patch: Any, history: Any, basis: Any,
            placement: Any, exit_module: Any, rows: list[Any]) -> None:
    from src import puyo_core_bridge as core
    control, cls = factory.controller, type(factory.controller)
    v1 = cls.prepared.__globals__['V1']
    lifecycle = v1.L
    old_hand, old_prepare = cls.hand, v1.prepared
    old_call, old_consume = cls.call, cls.consumed_history
    def hand(self: Any, binding: Any, view: Any) -> Any:
        record = getattr(binding, 'private_suffix_basis', None)
        if record is not None and getattr(binding, 'private_suffix_placement', None) is None:
            if not basis.start(self, binding, view):
                return None
        return old_hand(self, binding, view)
    def prepared(self: Any, binding: Any, item: Any, sm: Any, raw: Any, signals: Any, view: Any) -> Any:
        if (getattr(binding, 'private_suffix_basis', None) is not None
            and getattr(binding, 'private_suffix_placement', None) is None):
            return placement.prepare(basis, lifecycle, core, self, binding, item, sm, signals, view)
        return old_prepare(self, binding, item, sm, raw, signals, view)
    def call(self: Any, caller: Any, binding: Any, view: Any, pipe: Any, side: str, proposal: Any) -> Any:
        if proposal is not None and proposal.get('kind') == placement.KIND:
            return self._parts.C.Controller.call(self, caller, binding, view, pipe, side, proposal)
        return old_call(self, caller, binding, view, pipe, side, proposal)
    def consumed(self: Any, value: Any, caller: Any) -> None:
        if value['prepared'].get('kind') == placement.KIND:
            return placement.consumed(basis, self, value, caller, rows)
        return old_consume(self, value, caller)
    for obj, name, value in ((cls, 'hand', hand), (v1, 'prepared', prepared),
        (cls, 'call', call), (cls, 'consumed_history', consumed)):
        patch(stack, obj, name, value)
    tail_capture(stack, history, patch, basis)
    exit_module.install(stack, patch, basis, placement)
