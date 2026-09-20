"""空tail→後発原NEXTの保存連続性だけを検査。原J/pop/Sは実行・変更しない。"""
from __future__ import annotations
from typing import Any

KIND = 'hidden_empty_tail_next_history/v1'
TAIL = 'hidden_single_tail_history/v1'
CAPTURE = 'empty_tail_captured/v1'
START = 'empty_tail_new_head_started/v1'
SCOPE_SIZE, PAIR_SIZE = 7, 2
PROOF_KEYS = {'kind','token','pair','occurred','available_frame','available_time','clear_last',
    'clear_observations','available_window','new_token','observed_raw','raw_capture','inferred_final',
    'previous_private_placement','previous_private_frame','prefix_source','inferred_grid_is_observed',
    'physical_certified','current_permission','scope','before_grid','grid'}


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError('empty_tail_stage1:'+reason)


def thaw(value: Any) -> Any:
    """元SP.contentの型タグを検査して復号する。bool/intの無条件変換はしない。"""
    require(type(value) in (tuple,list) and len(value) == PAIR_SIZE, 'content_shape')
    kind, data = value
    if kind == 'dict':
        require(type(data) in (tuple,list), 'content_dict')
        require(all(type(v) in (tuple,list) and len(v) == PAIR_SIZE and type(v[0]) is str
                    for v in data), 'content_entries')
        keys = [v[0] for v in data]
        require(keys == sorted(set(keys)), 'content_duplicate_order')
        return {key:thaw(item) for key,item in data}
    if kind in ('tuple','list'):
        require(type(data) in (tuple,list), 'content_sequence')
        result = [thaw(item) for item in data]
        return tuple(result) if kind == 'tuple' else result
    types = {'str':str, 'int':int, 'float':float, 'bool':bool, 'NoneType':type(None)}
    require(kind in types and type(data) is types[kind], 'content_scalar_type')
    return data


def scoped(audit: Any, record: Any) -> Any:
    e, scope = audit.m.E, record['scope']
    require(type(scope) in (tuple,list) and len(scope) == SCOPE_SIZE, 'scope7')
    require(all(type(scope[i]) is int and scope[i] >= 0 for i in (2,3,4,5)), 'scope_types')
    frame = record['frame']
    require(type(frame) is int and type(record['clock']) is float and record['clock'] == frame/e.FPS,
            'record_clock')
    row = audit.rows[frame]
    step = audit.step(row)
    e.same([scope[i] for i in (0,1,2,3,5,6)], [step['source_id'],step['run_id'],step['software_reset'],
        step['pipe_object_id'],step['generation']['reset_epoch'],step['side']], 'empty_saved_scope')
    require(step['status'] == 'returned' and step['exception'] is None, 'step_return')
    require(record['current_permission'] is record['physical_certified'] is False, 'record_permission')
    require(type(record['binding_id']) is int and record['binding_id'] > 0, 'binding_id')
    e.same(record['state'], row['decision']['history_state'], 'empty_snapshot_state')
    return row


def policy(audit: Any, capture: Any) -> None:
    e, state = audit.m.E, capture['state']
    values = thaw(capture['policy_content'])
    require(type(values) is list and bool(values), 'policy_list')
    entries = state['history']
    require(len(values) == len(entries)+1, 'policy_history_coverage')
    baseline = values[0]
    require(set(baseline) == {'purpose','proof','proof_sha'} and baseline['purpose'] == 'baseline'
            and baseline['proof_sha'] == e.digest(baseline['proof']), 'baseline_policy')
    initial = baseline['proof']
    e.same(initial['scope'], state['scope'], 'baseline_scope')
    e.same(e.counts(initial['grid']), state['baseline'], 'baseline_counts')
    require(initial['frame'] == e.clock(state['baseline_through'], 0), 'baseline_frame')
    e.same(initial['grid'], audit.rows[initial['frame']]['baseline_grid'], 'baseline_original_grid')
    expected = dict(kind='live_history_baseline',frame=initial['frame'],time_sec=initial['frame']/e.FPS,
        grid=initial['grid'],live_scope=capture['scope'],prior_legacy_debt='UNKNOWN',scope=state['scope'])
    e.same(initial, expected, 'baseline_original_fixed_fields')
    require(state['scope']['game_id'] == 'live-private-NEXT-interval:'+e.digest([capture['scope'],initial['frame']]),
            'baseline_original_scope_digest')
    for value, entry in zip(values[1:], entries):
        require(type(value) is dict and set(value) == {'purpose','proof','proof_sha'}, 'policy_fields')
        p = value['proof']
        require(value['proof_sha'] == e.digest(p) == entry['event_id'], 'policy_event_digest')
        require(value['purpose'] == entry['kind'], 'policy_purpose')
        require(value['purpose'] == 'placement', 'unsupported_prior_policy')
        e.same(p, audit.rows[entry['available_at']['frame']]['prepared'], 'policy_original_proof')
    require(values[-1]['purpose'] == 'placement', 'tail_last_policy')
    e.same(values[-1]['proof'], capture['proof'], 'tail_policy_proof')


def tail_source(audit: Any, capture: Any, tail: Any) -> None:
    e, p = audit.m.E, tail['prepared']
    frame = p['prefix_consumed_frame']
    prefix = audit.rows[frame]['prepared']
    require(prefix['kind'] == 'hidden_two_hand_prefix_history/v1', 'tail_prefix_kind')
    e.same(p['prefix_source'], prefix['directional_commit'], 'tail_prefix_source')
    e.same(thaw(capture['source_content']), (prefix['directional_commit'],prefix['inferred_path'],
        prefix['occurred'],prefix['clear_last'],prefix['clear_observations']), 'tail_source_full')
    require(prefix['new_token'] == p['token'] and frame < p['occurred'][0], 'tail_prefix_token_time')
    e.same(prefix['grid'], p['before_grid'], 'tail_prefix_before')
    e.same(prefix['inferred_path']['final'], p['grid'], 'tail_support_final')
    votes = thaw(capture['votes_content'])
    require(type(votes) is tuple and len(votes) == 12, 'tail_votes_shape')
    e.same(votes[:9], (capture['scope'],p['token'],p['grid'],p['observed_raw'],p['raw_capture'],
        p['occurred'],p['clear_last'],p['clear_observations'],prefix['grid']), 'tail_votes_full')
    require(votes[-1] == frame, 'tail_item_started')
    for pair in votes[9:11]:
        require(type(pair) is tuple and len(pair) == PAIR_SIZE
            and all(type(c) is int and c in e.COLORS for c in pair), 'tail_item_pair')
    enqueue = audit.enqueues[(frame,tail['scope']['side'])]
    e.same(votes[9], enqueue['after']['last_seen_next'], 'tail_item_NEXT')
    policy(audit, capture)


def captured(audit: Any, capture: Any) -> Any:
    e, row = audit.m.E, scoped(audit, capture)
    p, step = row['prepared'], audit.step(row)
    require(type(p) is dict and p['kind'] == TAIL and row['decision']['history_consumed'] is True,
            'capture_original_tail')
    require(capture['head'] is capture['token'] is row['next_token'] is None and not row['added'],
            'tail_future_head')
    e.same(capture['proof'], p, 'capture_original_proof')
    require(capture['digest'] == e.digest(p), 'capture_digest')
    e.same(capture['grid'], p['grid'], 'capture_grid')
    e.grid(capture['old_current'])
    e.same(capture['old_current'], row['returned']['grid'], 'capture_old_confirmed')
    token, scope, events = thaw(capture['journal_content'])
    require(token == step['token'], 'capture_step_token')
    e.same(scope, row['scope'], 'capture_J_scope')
    actual = [v for v in step['events'] if v['stage'] == 'fifo_after']
    e.same(events, actual, 'capture_original_J_events')
    after = e.singleton(actual, 'empty_tail_after')
    before = e.singleton([v for v in step['events'] if v['stage'] == 'fifo_before'], 'empty_tail_before')
    e.same(before['fifo_occurrence_tokens'], [p['token']], 'tail_single_token')
    e.same(before['accounting']['pending_tsumo'], [p['pair']], 'tail_single_pair')
    require(after['accounting']['pending_tsumo'] == [], 'tail_empty_after')
    require(not capture['state']['origins'] and not capture['state']['debts'], 'capture_no_origin')
    tail_source(audit, capture, row)
    return row


def started(audit: Any, capture: Any, start: Any, journal: Any) -> Any:
    e, row = audit.m.E, scoped(audit, start)
    e.same(start['scope'], capture['scope'], 'start_scope7')
    require(start['binding_id'] == capture['binding_id'] and start['old_tail_digest'] == capture['digest'],
            'start_basis_identity')
    require(capture['frame'] < start['frame'] and row['prepared'] is None, 'start_after_tail')
    token = start['token']
    require(type(token) is str and token != capture['proof']['token'], 'future_token')
    e.same(row['view_tokens'], [token], 'start_single_head')
    e.same(row['added'], [token], 'start_added')
    require(row['next_token'] == token, 'start_H_token')
    step = audit.step(row)
    enqueue = journal.created(audit, token, step)
    require(enqueue['frame_idx'] == start['frame'], 'start_creation_samecall')
    for key, expected in (('added_occurrence_tokens',[token]),('fifo_occurrence_tokens',[token]),
                          ('discarded_tokens',[])):
        e.same(enqueue[key], expected, 'empty_enqueue:'+key)
    e.same(enqueue['before']['pending_tsumo'], [], 'enqueue_was_empty')
    e.same(enqueue['after']['pending_tsumo'], [start['head']], 'enqueue_new_pair')
    e.same(enqueue['after']['last_consumed_color'], start['head'], 'enqueue_head_value')
    b = thaw(start['bound_content'])
    require(type(b) is tuple and len(b) == 8, 'bound_shape')
    e.same(b[:5], (token,(start['frame'],start['clock']),token,start['scope'],capture['grid']), 'bound_source')
    e.same(b[5], enqueue['pair'], 'bound_original_NEXT')
    require(b[7] == start['frame'] and type(b[6]) is tuple and len(b[6]) == PAIR_SIZE
        and all(type(c) is int and c in e.COLORS for c in b[6]), 'bound_DNEXT_started')
    before, after = capture['state'], start['state']
    require(type(after['action']) is int and after['action'] == before['action']+1, 'new_action')
    for key in set(before)-{'clock','action','action_since'}:
        e.same(before[key], after[key], 'start_state_unchanged:'+key)
    require(e.clock(after['clock'], 2) == e.clock(after['action_since'], 2) == start['frame'], 'start_clock')
    return row


def continuous(audit: Any, capture: Any, start: Any, placed: Any) -> None:
    e, end = audit.m.E, placed['scope']['frame_idx']
    for frame in range(capture['frame']+e.STRIDE, end+e.STRIDE, e.STRIDE):
        row = audit.rows[frame]
        step = audit.step(row)
        require(step['status'] == 'returned' and step['exception'] is None, 'gap_step_return')
        e.same(row['decision']['history_state']['scope'], capture['state']['scope'], 'gap_owner_scope')
        if frame < start['frame']:
            require(not row['view_tokens'] and not row['added'] and row['next_token'] is None
                and row['prepared'] is None, 'empty_gap')
            e.same(row['decision']['history_state'], capture['state'], 'empty_gap_state')
        else:
            e.same(row['view_tokens'], [start['token']], 'bound_queue_continuity')
            e.same(row['added'], [start['token']] if frame == start['frame'] else [], 'bound_no_extra_added')
            if frame < end:
                require(row['prepared'] is None and row['next_token'] == start['token'], 'bound_no_earlier_pop')
                e.same(row['decision']['history_state'], start['state'], 'bound_old_state')
        if frame < end:
            require(not row['decision']['history_consumed'], 'no_intermediate_consume')


def placed(audit: Any, capture: Any, start: Any, row: Any, check: Any) -> dict[str, Any]:
    e, p = audit.m.E, row['prepared']
    require(set(p) == PROOF_KEYS and p['kind'] == KIND, 'new_proof_fields_kind')
    require(p['previous_private_placement'] == capture['digest']
        and p['previous_private_frame'] == capture['frame'] and p['token'] == start['token'], 'new_basis_link')
    e.same(p['before_grid'], capture['grid'], 'new_basis_grid')
    e.same(p['scope'], capture['state']['scope'], 'new_basis_scope')
    e.same(p['prefix_source'], capture['proof']['prefix_source'], 'new_prefix_source')
    e.same(p['pair'], start['head'], 'new_head_pair')
    e.same(p['grid'], p['inferred_final'], 'new_final_grid')
    require(p['available_window'] is False and p['new_token'] is None, 'new_proof_window_token')
    require(start['frame'] < p['occurred'][0] and type(p['clear_observations']) is int
            and p['clear_observations'] == PAIR_SIZE, 'new_fresh_two')
    require(p['inferred_grid_is_observed'] is p['physical_certified'] is p['current_permission'] is False,
            'new_permission')
    expected = dict(frame=row['scope']['frame_idx'],token=p['token'],event_id=e.digest(p),queue_slots=1,
        state_action=start['state']['action'],current_permission=False)
    e.same(check, expected, 'prepop_proof_receipt')
    continuous(audit, capture, start, row)
    return dict(tail_frame=capture['frame'], start_frame=start['frame'], frame=p['available_frame'],
        token=p['token'], event_id=e.digest(p), original_empty_FIFO_verified=True)


def verify(audit: Any, evidence: Any, journal: Any) -> dict[str, Any]:
    require(type(evidence) is dict and set(evidence) == {'basis_events','prepop_checks',
        'output_identity_is_not_live_authorization','physical_certified','quality_gate_clear'}, 'evidence_required')
    require(evidence['output_identity_is_not_live_authorization'] is True
        and evidence['physical_certified'] is evidence['quality_gate_clear'] is False, 'evidence_authority')
    rows = [r for r in audit.rows.values() if r['prepared'] and r['prepared']['kind'] == KIND]
    events, checks = evidence['basis_events'], evidence['prepop_checks']
    require(type(events) is type(checks) is list and len(events) == len(rows)*PAIR_SIZE
        and len(checks) == len(rows) and bool(rows), 'empty_evidence_coverage')
    used, report = set(), []
    for row in rows:
        p = row['prepared']
        capture = audit.m.E.singleton([v for v in events if v['kind'] == CAPTURE
            and v['frame'] == p['previous_private_frame']], 'empty_capture_once')
        start = audit.m.E.singleton([v for v in events if v['kind'] == START
            and v['old_tail_digest'] == capture['digest']], 'empty_start_once')
        require(capture['frame'] not in used, 'basis_reuse')
        used.add(capture['frame'])
        captured(audit, capture)
        started(audit, capture, start, journal)
        check = audit.m.E.singleton([v for v in checks if v['frame'] == p['available_frame']], 'prepop_once')
        report.append(placed(audit, capture, start, row, check))
    require(all(v['kind'] in (CAPTURE,START) for v in events), 'unknown_basis_event')
    return dict(links=report, saved_structure_verified=True, actual_scope4_identity_verified=False,
        DNext_live_value_reauthorized=False, original_Link_quiet_reauthorized=False,
        samecall_verified=False, native_pop_replayed=False, integer_writer_called=False)
