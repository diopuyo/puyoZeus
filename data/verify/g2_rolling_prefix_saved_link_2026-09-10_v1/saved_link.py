"""複数prefixの完全連結だけを監査する。既存Stage1/worldの代替ではない。"""
from __future__ import annotations
from typing import Any
import rolling_link_fixed as F
import rolling_link_journal as J


def groups(e: Any, rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    result, targets = {}, set()
    for row in rows:
        value = row.get('decision', {}).get('history_state')
        if value is None:
            F.require(not F.prefix(row), 'prefix_without_state')
            continue
        key = e.encoded(value['scope'])
        result.setdefault(key, []).append(row)
        if F.prefix(row):
            targets.add(key)
    return [values for key, values in result.items() if key in targets]


def coverage(e: Any, rows: list[dict[str, Any]]) -> dict[int, Any]:
    by_frame, retained, seen = {}, [], {}
    scope = F.state(rows[0])['scope']
    for row in rows:
        frame = row['scope']['frame_idx']
        F.require(frame not in by_frame and (not by_frame or frame > next(reversed(by_frame))), 'history_order')
        by_frame[frame] = row
        state = e.state(row, scope)
        history = state['history']
        e.same(history[:len(retained)], retained, 'rolling_retained_history_prefix')
        F.require(len(history) >= len(retained), 'history_shrink')
        for entry in history:
            if entry['kind'] != F.PLACEMENT:
                continue
            event_frame = entry['available_at']['frame']
            F.require(event_frame in by_frame, 'predecessor_row_missing')
            actual = by_frame[event_frame]
            e.same(F.event(e, actual), entry, 'rolling_retained_event')
            seen[event_frame] = entry
        if row['decision']['history_consumed'] is True:
            F.event(e, row)
        retained = history
    return seen


def proof(e: Any, row: Any, prior: Any) -> Any:
    p, state = row['prepared'], F.state(row)
    entry = F.event(e, row)
    e.same(p['scope'], state['scope'], 'rolling_proof_scope')
    e.same(e.grid(p['grid']), e.grid(row['grid_after']), 'rolling_private_grid')
    e.grid(p['before_grid'])
    support = p['inferred_path']
    F.require(type(support) is dict and set(support) == F.SUPPORT_KEYS, 'support_fields')
    e.same(support['prefix'], p['grid'], 'rolling_support_prefix')
    e.same(support['raw'], p['observed_raw'], 'rolling_support_raw')
    e.same(support['pairs'][0], p['pair'], 'rolling_support_pair')
    for flag in ('physical_certified', 'probability_assigned', 'accounting_permission', 'current_permission'):
        F.require(support[flag] is False, 'support_authority')
    F.require(p['current_permission'] is p['physical_certified'] is p['intermediate_grid_is_observed'] is False,
              'prefix_authority')
    F.require(row['decision']['current_permission'] is False and row['decision'].get('current_proof') is None,
              'integer_authority')
    F.require(prior is not None, 'prefix_before_row_missing')
    e.same(p['before_grid'], prior['grid_after'], 'rolling_exact_before_grid')
    previous_slot = F.state(prior)['current']
    expected_slot = None if previous_slot is None else dict(previous_slot, available=False)
    e.same(state['current'], expected_slot, 'rolling_old_integer_slot')
    e.same(entry['action'], F.state(prior)['action'], 'rolling_placement_action')
    F.require(state['action'] == entry['action'] + 1, 'prefix_next_action')
    first, last = F.point(e, p['occurred']), F.point(e, p['clear_last'])
    F.require(type(p['clear_observations']) is int and p['clear_observations'] >= 2, 'vote_count')
    F.require(first[0] + (p['clear_observations'] - 1) * e.STRIDE == last[0], 'vote_adjacent')
    e.same(list(last), [row['scope']['frame_idx'], row['scope']['time_sec']], 'rolling_vote_last')
    e.same(list(last), [p['available_frame'], p['available_time']], 'rolling_available_clock')
    consume, commit = p['directional_consumption'], p['directional_commit']
    expected = dict(kind='directional_deferred_consumption/v1', source_available_frame=commit['available_frame'],
        source_available_time=commit['available_time'], consumer_frame=last[0], consumer_time=last[1],
        old_token=p['token'], new_token=p['new_token'], original_directional_sha256=e.digest(commit),
        physical_certified=False)
    e.same(consume, expected, 'rolling_consumption_provenance')
    return entry


def linked(e: Any, old: Any, new: Any) -> dict[str, Any]:
    a, b = old['prepared'], new['prepared']
    before, after = F.event(e, old), F.event(e, new)
    e.same(b['before_grid'], a['grid'], 'rolling_predecessor_grid')
    e.same(b['token'], a['new_token'], 'rolling_predecessor_token')
    e.same(b['scope'], a['scope'], 'rolling_predecessor_scope')
    F.require(after['action'] == before['action'] + 1, 'predecessor_action')
    F.require(a['available_frame'] < b['directional_commit']['available_frame'] <= b['occurred'][0],
              'predecessor_source_time')
    F.require(a['directional_commit']['journal_call_token'] != b['directional_commit']['journal_call_token'],
              'predecessor_source_reuse')
    e.same(a['directional_commit']['segment_id'], b['directional_commit']['segment_id'], 'rolling_segment')
    e.same(F.state(new)['history'][:len(F.state(old)['history'])], F.state(old)['history'], 'rolling_predecessor_history')
    J.scope(e, dict(old['scope'], status='returned', software_reset=old['scope']['generation']['reset_epoch']),
            dict(new['scope'], status='returned', software_reset=new['scope']['generation']['reset_epoch']))
    return dict(predecessor_frame=old['scope']['frame_idx'], frame=new['scope']['frame_idx'],
                predecessor_event_id=before['event_id'], event_id=after['event_id'],
                predecessor_event_sha256=e.digest(before), source_sha256=e.digest(b['directional_commit']),
                token=b['token'], new_token=b['new_token'], scope=b['scope'])


def check(history_rows: list[dict[str, Any]], journal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """同runの全保存履歴/Jを受け、旧validatorの前段へ置く権限なしの連結検査。"""
    e = F.libraries()
    grouped = groups(e, history_rows)
    report = dict(kind='rolling_prefix_saved_link/v1', applicable=bool(grouped), prefixes=0, rolling_links=[],
        saved_link_verified=False, original_validators_still_required=True, content_sha_is_signature=False,
        physical_certified=False, current_permission=False, runtime_permission=False, quality_gate_clear=False)
    if not grouped:
        return report
    ix = J.index(journal_rows)
    for rows in grouped:
        coverage(e, rows)
        prior, placed = None, None
        for row in rows:
            if F.prefix(row):
                proof(e, row, prior)
                J.source(e, ix, row)
                report['prefixes'] += 1
                if placed is not None and F.prefix(placed):
                    report['rolling_links'].append(linked(e, placed, row))
            if row['decision']['history_consumed'] is True:
                placed = row
            prior = row
    report['saved_link_verified'] = True
    return report
