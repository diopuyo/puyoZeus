"""第2段の保存内容結合。実参照のない保存検査をruntime完了と呼ばない。"""
from __future__ import annotations
import json
from typing import Any
import world_api as A
import world_geometry as G
import world_pb as B
import world_coverage as C


def hidden_histories(lib: Any, rows: Any, prepared: Any, histories: Any) -> None:
    expected = {key for key, value in prepared.items() if value['kind'] in (G.PREFIX, G.TAIL, G.CONTINUE)}
    joined = set()
    for row in rows:
        key = row['frame'], row['state']['scope']['side']
        B.require(key in expected and key not in joined, 'hidden_history_join')
        source, history = prepared[key], histories[key]
        B.equal(lib, row['source'], source, 'hidden_history_actual_source')
        B.equal(lib, row['state'], history['decision']['history_state'], 'hidden_history_actual_state')
        B.equal(lib, row['old_current'], history['returned']['grid'], 'hidden_history_SM_unchanged')
        B.require(row['kind'] == source['kind'] and row['current_permission'] is False
            and row['physical_certified'] is False and row['native_counter_unchanged'] is True, 'hidden_history_flags')
        name = 'inferred_prefix' if source['kind'] == G.PREFIX else 'inferred_final'
        B.equal(lib, row[name], source['grid'], 'hidden_history_world')
        joined.add(key)
    B.require(joined == expected, 'missing_hidden_history')


def current_sources(lib: Any, currents: Any, prepared: Any, origins: Any, settled: Any) -> None:
    for (frame, side), value in currents.items():
        proof = json.loads(value['evidence_json'])
        available = [(key, source) for key, source in prepared.items() if key[1] == side and key[0] <= frame]
        B.require(bool(available), 'current_without_world')
        _, source = max(available, key=lambda item: item[0][0])
        if proof['kind'] == 'conditional_hidden_current/v1':
            B.require(source['kind'] in (G.TAIL, G.CONTINUE), 'current_history_kind')
            B.equal(lib, value['inferred_grid'], source['grid'], 'current_history_world')
            B.equal(lib, proof['prefix_source'], source['prefix_source'], 'current_prefix_source')
            first = prepared[proof['prefix_consumed_frame'], side]
            tail = prepared[proof['tail_consumed_frame'], side]
            B.require(first['kind'] == G.PREFIX and tail['kind'] == G.TAIL, 'current_initial_path')
            B.equal(lib, first['directional_commit'], proof['prefix_source'], 'current_original_direction')
            continue
        origin = proof['conditional_origin']
        key = origin['evidence']['origin_id']
        actual, saved_origin = origins[key]
        B.equal(lib, origin, saved_origin, 'current_actual_origin')
        B.equal(lib, proof['conditional_settlement'], settled[key], 'current_actual_settlement')
        B.require(settled[key]['evidence']['available_at']['frame'] <= frame, 'current_after_settlement')
        if proof['kind'] == 'conditional_hidden_after_firing_next_current/v1':
            B.require(source['kind'] == G.NEXT, 'current_after_next_kind')
            B.equal(lib, proof['next_placement'], source, 'current_actual_next')
            B.equal(lib, value['inferred_grid'], source['grid'], 'current_next_world')
        else:
            B.require(source['kind'] == G.FIRING, 'current_after_firing_kind')
            B.equal(lib, value['inferred_grid'], actual.predicted_final, 'current_settled_world')


def lifetime(lib: Any, rows: Any, currents: Any, histories: Any) -> None:
    seen = set()
    for row in rows:
        frame = row['frame']
        B.require(row['kind'] == 'candidate_retired' and frame not in seen, 'lifetime_kind_or_duplicate')
        prior = [(key, v) for key, v in currents.items() if key[0] < frame]
        B.require(bool(prior), 'retire_without_current')
        key, value = max(prior, key=lambda item: item[0][0])
        current_key = frame, key[1]
        history = histories[current_key]
        B.require(current_key not in currents and row['anchor_retained'] is True, 'retire_current_or_anchor')
        B.require(row['action'] == history['decision']['history_state']['action']
            and row['pending'] == bool(history['view_tokens']), 'retire_actual_context')
        B.equal(lib, value['integer_anchor'], lib.C.encoded(history['decision']['history_state']['current']),
            'retire_integer_slot_retained')
        seen.add(frame)


def verify_world(*, history_rows: Any, journal_rows: Any, conditional_rows: Any,
                 hidden_events: Any, hidden_history: Any, hidden_lifetime: Any,
                 outer_rows: Any, controller: Any = None, factory: Any = None) -> dict[str, Any]:
    result = A.verify_probabilities(history_rows=history_rows, journal_rows=journal_rows,
        hidden_events=hidden_events, outer_rows=outer_rows, controller=controller, factory=factory)
    histories = A.indexed(history_rows, lambda row: (row['scope']['frame_idx'], row['scope']['side']))
    currents = {(row['frame'], row['current']['scope'][-1]): row['current']
        for row in hidden_events if row.get('kind') == B.EVENT}
    with G.libraries() as ctx:
        prepared = G.histories(ctx, history_rows, currents)
        outer = A.indexed(outer_rows, lambda row: row['frame_idx'])
        calls, expected = None, None
        if factory is not None:
            join = G.L.load('_world_original_capture_join', G.L.VERIFY /
                'g2_conditional_current_call_join_2026-09-10_v1/call_join.py')
            calls = join.verify(ctx.lib, controller, factory, histories, journal_rows, currents)
            expected = {tuple(key) for key in calls['expected_current_keys']}
        C.verify(ctx.lib, histories, journal_rows, currents, prepared, outer, expected)
        origins = G.origins(ctx, conditional_rows, prepared, histories)
        settled = G.settlements(ctx, conditional_rows, origins, histories)
        hidden_histories(ctx.lib, hidden_history, prepared, histories)
        current_sources(ctx.lib, currents, prepared, origins, settled)
        lifetime(ctx.lib, hidden_lifetime, currents, histories)
        result.update(geometry_verified=True, backend=ctx.backend, prepared_worlds=len(prepared),
            origins_verified=len(origins), settlements_verified=len(settled), lifetime_rows=len(hidden_lifetime),
            world_PB_verified=True, stage1_required=True, runtime_finalization_allowed=False,
            original_capture_calls=calls, candidate_coverage_mode='saved_conservative_qualification'
            if calls is None else 'original_capture_return_and_publication')
    return result
