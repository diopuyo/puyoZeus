"""PB内容結合の段階別入口。geometryの未検証をPASSへ含めない。"""
from __future__ import annotations
from typing import Any
import world_libs as L
import world_pb as B


def indexed(rows: Any, key: Any) -> dict[Any, Any]:
    result = {key(row): row for row in rows}
    B.require(len(result) == len(rows), 'duplicate_rows')
    return result


def verify_probabilities(*, history_rows: Any, journal_rows: Any, hidden_events: Any,
                         outer_rows: Any, controller: Any = None, factory: Any = None) -> dict[str, Any]:
    lib = L.libraries()
    B.require((controller is None) == (factory is None), 'live_context_pair')
    if factory is not None:
        B.require(controller is factory.controller and hidden_events is controller.hidden_current_events,
            'actual_controller_events')
    history = indexed(history_rows, lambda row: (row['scope']['frame_idx'], row['scope']['side']))
    journal = indexed([row for row in journal_rows if row['kind'] == 'step'], lambda row: row['token'])
    outer = indexed(outer_rows, lambda row: row['frame_idx'])
    events = [row for row in hidden_events if row.get('kind') == B.EVENT]
    B.require(bool(events), 'no_conditional_candidates')
    B.require(all(row.get('kind') in (B.EVENT, 'conditional_candidate_hold', 'natural_hidden_exit_preview')
        for row in hidden_events), 'unknown_hidden_event')
    indexed(events, lambda row: (row['frame'], row['current']['scope'][-1]))
    checked, kinds, live_scopes = [], set(), []
    for event in events:
        value, proof = B.certificate(lib, event)
        row = history[(value.frame, value.scope[-1])]
        step = journal[row['journal_token']]
        live_scopes.append(B.scope(lib, value, row, step, factory))
        B.grids(lib, value, proof, row)
        B.original(lib, value, proof, row, step, outer[value.frame])
        checked.append(value.frame); kinds.add(proof['kind'])
    return dict(conditional_frames=checked, kinds=sorted(kinds), probability_content_verified=True,
        original_PB_preserved=True, actual_live_scope_verified=bool(events) and all(live_scopes),
        geometry_verified=False, runtime_finalization_allowed=False, physical_certified=False,
        integer_current_permission=False, quality_gate_clear=False)
