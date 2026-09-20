"""原captureを一回委譲し、make実返却を同producerの別ledgerへ記録する。"""
from __future__ import annotations
from typing import Any
import current_call_fixed as F
import current_call_values as V

ROWS = 'conditional_current_call_rows'
REFS = 'conditional_current_call_refs'


def wrapped(factory: Any, module: Any, control: Any, journal: Any, original: Any) -> Any:
    rows, refs, tokens = getattr(control, ROWS), getattr(control, REFS), set()
    def capture(actual: Any, recorder: Any, item: Any, result: Any, provisional: Any) -> Any:
        returned, completed = None, False
        try:
            F.require(factory.controller is control and factory.provider.journal is journal
                and actual is control and recorder is journal, 'actual_factory_changed')
            F.require(module.capture is capture and original.__globals__ is vars(module), 'capture_reference')
            returned = original(actual, recorder, item, result, provisional)
            completed = True
            row = V.success(module, control, journal, item, returned)
            F.require(row['token'] not in tokens, 'duplicate_token')
            tokens.add(row['token'])
            if returned[1] is not None:
                refs[row['token']] = returned[1]
            rows.append(row)
            return returned
        except BaseException as error:
            rows.append(V.failed(item, error, returned, completed))
            raise
    return capture


def install(stack: Any, factory: Any, patch: Any) -> Any:
    module, control, journal, original = F.original(factory)
    F.require(not hasattr(control, ROWS) and not hasattr(control, REFS), 'duplicate_install')
    F.require(journal.steps == 0, 'install_before_updates')
    setattr(control, ROWS, [])
    setattr(control, REFS, {})
    patch(stack, module, 'capture', wrapped(factory, module, control, journal, original))
    return original


def references(factory: Any) -> tuple[Any, ...]:
    module, _, journal = F.select(factory)
    return module.capture, journal.complete_step
