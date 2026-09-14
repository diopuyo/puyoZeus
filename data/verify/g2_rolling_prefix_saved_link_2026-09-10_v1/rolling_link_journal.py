"""元Jの発生票と一回popを、保存prefix sourceへ結ぶ。token文字列は解析しない。"""
from __future__ import annotations
from typing import Any
from types import SimpleNamespace as N
from rolling_link_fixed import generation_library, point, require

J_SCOPE = ('source_id', 'run_id', 'side', 'software_reset', 'pipe_object_id')
GENERATION = ('identity_scope', 'reset_epoch', 'side')
NATIVE = ('tsumo_count', 'first_move_sec', 'landing_pending', 'last_consumed_color')
SOURCE_KEYS = {'kind', 'source_id', 'run_id', 'side', 'software_epoch', 'segment_id',
    'available_frame', 'available_time', 'old_token', 'new_token', 'journal_call_token',
    'candidate', 'old_accepted', 'new_accepted', 'dnext', 'physical_progress_certified'}
CANDIDATE_KEYS = {'sequence_number', 'first_support_frame', 'available_frame', 'reference'}


def index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls, created, pops = {}, {}, {}
    for row in rows:
        if row['kind'] not in ('step', 'enqueue'):
            continue
        require(row['token'] not in calls, 'J_duplicate_call')
        calls[row['token']] = row
        for token in row.get('added_occurrence_tokens', []):
            created.setdefault(token, []).append(row)
        for value in row.get('events', []):
            if value['stage'] == 'fifo_after' and value.get('enqueue_occurrence_token') is not None:
                pops.setdefault(value['enqueue_occurrence_token'], []).append(row)
    return dict(calls=calls, created=created, pops=pops, generation_checked=set())


def scope(e: Any, before: dict[str, Any], after: dict[str, Any]) -> None:
    for key in J_SCOPE:
        e.same(before[key], after[key], 'rolling_J_scope:' + key)
    for key in GENERATION:
        e.same(before['generation'][key], after['generation'][key], 'rolling_generation:' + key)
    a, b = before['generation']['action_revision'], after['generation']['action_revision']
    require((a is None and (b is None or type(b) is int and b >= 1))
            or type(a) is type(b) is int and 0 <= a <= b, 'generation_revision')
    require(before['status'] == after['status'] == 'returned', 'J_status')
    require(point(e, [before['frame_idx'], before['time_sec']]) <=
            point(e, [after['frame_idx'], after['time_sec']]), 'J_order')


def creation(e: Any, ix: Any, token: str, step: Any, pair: Any) -> Any:
    require(type(token) is str and bool(token), 'token_type')
    row = e.singleton(ix['created'].get(token, []), 'rolling_token_creation')
    require(row['kind'] == 'enqueue', 'creation_kind')
    scope(e, row, step)
    if row['generation']['action_revision'] is None:
        unknown_generation(e, ix, row)
    tokens = row['fifo_occurrence_tokens']
    require(tokens.count(token) == 1, 'creation_fifo')
    e.same(row['after']['pending_tsumo'][tokens.index(token)], pair, 'rolling_created_pair')
    return row


def unknown_generation(e: Any, ix: Any, origin: Any) -> None:
    key = tuple(origin[k] for k in J_SCOPE)
    if key in ix['generation_checked']:
        return
    rows = [v for v in ix['calls'].values() if v['kind'] == 'step'
            and tuple(v[k] for k in J_SCOPE) == key and v['frame_idx'] >= origin['frame_idx']]
    by_frame = {v['frame_idx']: v for v in rows}
    require(len(by_frame) == len(rows) and bool(rows), 'generation_step_unique')
    # Bindingではなく原J値の読取adapter。chainは元None→1/単調/全strideを検査する。
    audit = N(m=N(E=e), rows=by_frame, step=lambda value: value)
    generation_library().chain(audit, origin, max(rows, key=lambda value: value['frame_idx']))
    ix['generation_checked'].add(key)


def pop(e: Any, ix: Any, row: Any, step: Any) -> None:
    proof = row['prepared']
    before = e.singleton([v for v in step['events'] if v['stage'] == 'fifo_before'], 'rolling_fifo_before')
    after = e.singleton([v for v in step['events'] if v['stage'] == 'fifo_after'], 'rolling_fifo_after')
    tokens = [proof['token'], proof['new_token']]
    e.same(before['fifo_occurrence_tokens'], tokens, 'rolling_original_heads')
    e.same(row['view_tokens'], tokens, 'rolling_view_heads')
    e.same(before['accounting']['pending_tsumo'], proof['inferred_path']['pairs'], 'rolling_actual_pairs')
    e.same(after['accounting']['pending_tsumo'], before['accounting']['pending_tsumo'][1:], 'rolling_suffix')
    e.same(after['committed'], proof['pair'], 'rolling_actual_pop')
    e.same(after['enqueue_occurrence_token'], proof['token'], 'rolling_pop_token')
    require(len(ix['pops'].get(proof['token'], [])) == 1, 'pop_once')
    for key in NATIVE:
        e.same(before['accounting'][key], after['accounting'][key], 'rolling_native:' + key)
    e.same(row['next_token'], proof['new_token'], 'rolling_next_token')


def source(e: Any, ix: Any, row: Any) -> Any:
    proof, commit = row['prepared'], row['prepared']['directional_commit']
    require(set(commit) == SOURCE_KEYS and set(commit['candidate']) == CANDIDATE_KEYS, 'source_fields')
    step = ix['calls'][row['journal_token']]
    e.journal(row, step)
    require(step['status'] == 'returned', 'prefix_step_returned')
    require(commit['kind'] == 'directional_commit_original_J/v1', 'source_kind')
    tokens = [proof['token'], proof['new_token']]
    require(len(set(tokens)) == 2, 'distinct_heads')
    e.same([commit['old_token'], commit['new_token']], tokens, 'rolling_source_tokens')
    creation(e, ix, tokens[0], step, proof['pair'])
    enqueue = creation(e, ix, tokens[1], step, proof['inferred_path']['pairs'][1])
    require(enqueue['token'] == commit['journal_call_token'], 'source_call')
    e.same([commit['source_id'], commit['run_id'], commit['side'], commit['software_epoch']],
           [step['source_id'], step['run_id'], step['side'], step['software_reset']], 'rolling_source_scope')
    e.same([commit['available_frame'], commit['available_time']],
           [enqueue['frame_idx'], enqueue['time_sec']], 'rolling_source_clock')
    require(point(e, [commit['available_frame'], commit['available_time']]) <= point(e, proof['occurred']),
            'source_future')
    e.same(enqueue['added_occurrence_tokens'], [tokens[1]], 'rolling_source_added')
    e.same(enqueue['fifo_occurrence_tokens'], tokens, 'rolling_source_queue')
    e.same(enqueue['before']['pending_tsumo'], [proof['pair']], 'rolling_source_before')
    e.same(enqueue['after']['pending_tsumo'], proof['inferred_path']['pairs'], 'rolling_source_after')
    e.same(commit['old_accepted'], enqueue['before']['last_seen_next'], 'rolling_old_accepted')
    e.same(commit['new_accepted'], enqueue['after']['last_seen_next'], 'rolling_new_accepted')
    require(commit['physical_progress_certified'] is False, 'source_authority')
    pop(e, ix, row, step)
    return enqueue
