"""元constructor閉鎖後の実factoryを最終集計へ渡す。元finishの他判定は触らない。"""
from __future__ import annotations
import json
from typing import Any
import fusion as F

FACTORY_KEY = 'conditional_runtime_factory'


def evaluate(goals: Any, rows: Any, legal: Any, output: Any, state: Any) -> dict[str, Any]:
    original = state['repeated_firing_constructor']
    assert original['installed'] and original['closed'] and original['references_restored'], 'constructor_unclosed'
    factory = state[FACTORY_KEY]
    control = factory.controller
    assert control.provider is factory.provider and not control.calls and not control.tickets, 'actual_control_unclosed'
    state['combined_restored']()
    assert state['conditional_full_installs'] == state['conditional_revision_connection']['installs'] == 1
    assert state['combined_install']['configure_calls'] == 1
    journal = [json.loads(line) for line in (output / 'atomic_journal.jsonl').read_text().splitlines()]
    def outer() -> Any:
        return json.loads((output / 'POSTCOMMIT_CONSUMER_ROWS.json').read_text())
    return F.evaluate(goals, rows, legal, output, firing_rows=original['rows'], journal_rows=journal,
        conditional_rows=state['conditional_full_rows'], hidden_events=control.hidden_current_events,
        hidden_history=control.hidden_history_rows, hidden_lifetime=control.hidden_lifetime_rows,
        outer_rows=outer, controller=control, factory=factory)
