"""原transaction.inferの認証済み一段中継を解決し、元Jを検査する。"""
from __future__ import annotations
import sys
from typing import Any
import quarantine_v3 as V3

require = V3.require


class Guard(V3.Guard):
    def origin_caller(self, caller: Any, previous: Any, current: Any, pair: Any) -> Any:
        r = self.recovery
        if caller.f_code in r.journal.codes:
            return caller
        method = r.control.infer
        if caller.f_code is method.__func__.__code__:
            local, source = caller.f_locals, caller.f_back
            require(local.get('self') is r.control and local.get('caller') is source,
                    'transaction_caller_identity')
            require(source is not None and source.f_code in r.journal.codes, 'transaction_original_J')
            original, args = local.get('original'), local.get('args')
            require(getattr(original, '__self__', None) is self
                    and getattr(original, '__func__', None) is type(self).infer, 'transaction_infer_binding')
            require(type(args) is tuple and len(args) >= 3 and args[0] is previous
                    and args[1] is current and args[2] is pair, 'transaction_infer_arguments')
            require(r.control.tickets.get(id(source)) is None, 'transaction_ticket_during_recovery')
            return source
        item = r.journal.active
        require(item is None or item['scope']['side'] != '1P', 'unrecognized_infer_transport')
        return caller

    def infer(self, previous: Any, current: Any, pair: Any, *args: Any, **kwargs: Any) -> Any:
        caller = sys._getframe(1)
        if self.recovery.pending is not None:
            caller = self.origin_caller(caller, previous, current, pair)
        selected = None if self.waiting(caller) else pair
        return self.original(previous, current, selected, *args, **kwargs)


def install(stack: Any, recovery: Any, namespace: dict[str, Any]) -> Guard:
    original = namespace['infer_placement']
    value = Guard(recovery, original)
    wrapper = value.infer
    namespace['infer_placement'] = wrapper
    def restore() -> None:
        require(namespace['infer_placement'] is wrapper, 'restore_binding_changed')
        namespace['infer_placement'] = original
    stack.callback(restore)
    return value
