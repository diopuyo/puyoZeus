"""計算閉鎖とは別に、履歴/current/外側公開の実イベントを数える。"""
from __future__ import annotations
from typing import Any
import common as K

PUB_STATUS = 'HISTORY_CURRENT_PUBLICATION_STATUS.json'
OUTER_STATUS = 'POSTCOMMIT_CONSUMER_STATUS.json'
OUTER_ROWS = 'POSTCOMMIT_CONSUMER_ROWS.json'
OUTER_COMPARISON = 'POSTCOMMIT_CONSUMER_COMPARISON.json'


def current_proof(row: Any) -> Any:
    proof = row['decision'].get('current_proof')
    if row['decision']['current_permission'] is not True:
        K.require(proof is None, 'unpermitted_current_proof')
        return None
    K.require(type(proof) is dict and proof['kind'] == 'completed_history_current', 'current_proof_missing')
    K.require((proof['frame'], proof['clock']) == (row['scope']['frame_idx'], row['scope']['time_sec']),
        'current_proof_clock')
    K.require(proof['scope']['side'] == row['scope']['side'] and proof['grid'] == row['grid_after'], 'current_proof_grid')
    K.require(set(proof['channels']) == {'raw', 'sm', 'returned', 'probability'}
        and all(grid == proof['grid'] for grid in proof['channels'].values()), 'current_channel_mismatch')
    K.require(row['decision']['history_consumed'] is True and proof['placement'] == row['prepared'],
        'current_without_history_commit')
    return proof


def history(rows: list[Any], legal: Any) -> dict[str, Any]:
    K.require([r['scope']['frame_idx'] for r in rows] == list(range(K.HISTORY_FIRST, K.END, K.STRIDE)),
        'history_coverage')
    tokens, consumed, currents = set(), [], []
    for row in rows:
        decision, frame = row['decision'], row['scope']['frame_idx']
        K.require(row['scope']['side'] == '1P' and row['error'] is None, 'history_scope_or_error')
        K.require(decision['legacy_counter_changed'] is False and decision['infer_calls'] == 0,
            'history_native_gate_changed')
        K.require(row['counter_before'] == row['account_after']['tsumo_count']
            and row['first_move_before'] == row['account_after']['first_move_sec'], 'history_counter_writer')
        if decision['history_consumed']:
            proof, token = row['prepared'], decision['old_token']
            K.require(token not in tokens and proof['token'] == token and row['committed'] is not None,
                'history_duplicate_or_missing_token')
            K.require(proof['available_frame'] == frame and proof['new_token'] == row['next_token'], 'history_proof_clock')
            before, after = (tuple(tuple(c) for c in proof[n]) for n in ('before_grid', 'grid'))
            K.require(proof['grid'] == row['grid_after'] and legal(before, after, row['committed']), 'history_placement')
            tokens.add(token)
            consumed.append(dict(frame=frame, token=token, next_token=row['next_token'], proof=proof))
        proof = current_proof(row)
        if proof is not None:
            currents.append(proof)
    return dict(history_consumed=consumed, current_proofs=currents, current_event_observed=bool(currents),
        historical_clock_forced=False, old_current_none_goal_applied=False)


def publication(output: Any, current_count: int) -> dict[str, Any]:
    status, rows = K.read(output / PUB_STATUS), K.read(output / OUTER_ROWS)
    K.require(status['receiver_closed'] is True and status['receiver_errors'] == [], 'publication_receiver_unclosed')
    K.require([r['frame_idx'] for r in rows] == list(K.FRAMES), 'outer_consumer_coverage')
    K.require(all(r['time_sec'] == r['frame_idx'] / K.FPS and r['old_projection_join_verified'] is True
        and r['comparison_completed'] is True for r in rows), 'outer_consumer_join')
    issued = sum(r['tickets_this_update'] for r in rows)
    released = sum(r['released_this_update'] for r in rows)
    K.require(issued == status['issued'] == current_count and released == status['released'] <= issued,
        'publication_event_count')
    comparison = K.read(output / OUTER_COMPARISON)
    K.require(comparison['all_consumers_ready'] is True, 'outer_comparison_not_ready')
    return dict(issued=issued, released=released, outer_publication_observed=released > 0,
        changed_frames=[r['frame_idx'] for r in rows if r['changed_sides']],
        comparison_equal=comparison['equal'], actual_collector_append_verified=False,
        no_event_is_current_success=False, whole_writer_noninterference_certified=False)


def evaluate(rows: list[Any], legal: Any, output: Any) -> dict[str, Any]:
    value = history(rows, legal)
    value.update(publication(output, len(value['current_proofs'])))
    return value | dict(current_permission=False, quality_gate_clear=False, production_permission=False,
        physical_certified=False, baseline_adoption_repair_not_connected=False)
