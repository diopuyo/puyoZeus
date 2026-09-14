"""原world/capture本体を保ち、退役済み整数scopeと確率scopeを混同しない。"""
from __future__ import annotations
from contextlib import ExitStack
import json
import sys
from typing import Any
import probability_owner as P


def scope_authority(original: Any, state: Any, lease: Any, counts: Any,
                    factory: Any, supplied: Any, frame: int) -> None:
    """この原world経路は旧整数の捕捉専用。確率区間の代用にはしない。"""
    assert type(frame) is int and frame <= lease.empty_evidence.frame, 'probability_not_integer_world'
    old = lease.archive.binding
    assert tuple(supplied) == tuple(old.scope), 'retired_full_scope'
    P.retired_owner(original.R, state, lease, factory, old, old.owner.state,
                    old.owner.state, (frame, P.SIDE))
    counts['retired'] += 1


def verify(original: Any, factory: Any, state: Any, lease: Any) -> dict[str, Any]:
    """全原履歴/J/条件付き捕捉/実consumerを渡し、元world検査を省略しない。"""
    P.probability_authority(state, lease, factory)
    control, output = factory.controller, state['output']
    def read(name: str) -> list[Any]:
        return [json.loads(line) for line in (output / name).read_text().splitlines()]
    full, journal = read('directional_history.jsonl'), read('atomic_journal.jsonl')
    rows = [row for row in full if 'decision' in row]
    assert rows and all('decision' in row or row.get('kind') == 'reset_baseline_wait' for row in full)
    counts = dict(retired=0, active=0)
    def authority(given: Any, seen: Any, owner: Any, supplied: Any, frame: int) -> None:
        assert given is lease and seen is counts
        scope_authority(original, state, lease, counts, owner, supplied, frame)
    adapted = original.clone(original.adapted, scope_authority=authority)
    with ExitStack() as stack:
        previous = list(sys.path)
        stack.callback(sys.path.__setitem__, slice(None), previous)
        sys.path.insert(0, str(original.Q.FINAL))
        world = original.Q.module('_probability_finish_original_world', original.Q.FINAL / 'empty_world.py')
        with world.loaded() as old:
            result = adapted(world, old, lease, counts)(history_rows=rows, journal_rows=journal,
                conditional_rows=state['conditional_full_rows'], hidden_events=control.hidden_current_events,
                hidden_history=control.hidden_history_rows, hidden_lifetime=control.hidden_lifetime_rows,
                outer_rows=state['postcommit_publication_consumer'].rows, controller=control, factory=factory)
    assert result['world_PB_verified'] and result['actual_live_scope_verified']
    assert counts['retired'] > 0 and counts['active'] == 0
    return dict(world=result, scope_authority_calls=counts, old_world_verified=True,
                probability_world_verified=False, current_history_replaced=False,
                actual_factory=True, quality_gate_clear=False)
