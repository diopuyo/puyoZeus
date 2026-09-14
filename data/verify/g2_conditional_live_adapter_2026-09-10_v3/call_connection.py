"""既存共有設置の内側で原capture観測だけを所有し、先に復元する。"""
from __future__ import annotations
from typing import Any

STATE_KEY = 'conditional_current_call_observer'
FILENAME = 'CONDITIONAL_CURRENT_CALLS.json'


def closed(observer: Any, factory: Any, state: Any, before: Any, write: Any) -> None:
    receipt = state[STATE_KEY]
    restored = observer.references(factory) == before
    receipt.update(closed=True, references_restored=restored,
        rows=factory.controller.conditional_current_call_rows)
    write(state['output'] / FILENAME, receipt)
    assert restored, 'conditional_capture_not_restored'


def install(stack: Any, observer: Any, factory: Any, state: Any, patch: Any, write: Any) -> None:
    assert STATE_KEY not in state, 'conditional_capture_duplicate_install'
    before = observer.references(factory)
    state[STATE_KEY] = dict(installed=False, closed=False, references_restored=False,
        same_producer_as_outputs=True, physical_certified=False, quality_gate_clear=False)
    stack.callback(closed, observer, factory, state, before, write)
    observer.install(stack, factory, patch)
    state[STATE_KEY]['installed'] = True


def verify(state: Any) -> None:
    value = state[STATE_KEY]
    assert value['installed'] and value['closed'] and value['references_restored'], 'conditional_capture_unclosed'
    assert all(row['error'] is None for row in value['rows']), 'conditional_capture_failed'
