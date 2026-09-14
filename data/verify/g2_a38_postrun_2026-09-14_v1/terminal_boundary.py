"""到来台帳の試合終了票。未ACKを消費に変換せず、確率盤面更新権限も付与しない。"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any

FPS, STRIDE = 60, 2


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('terminal_boundary:' + reason)


def validate_scope(ledger: Any, row: dict) -> None:
    expected = ledger.scope
    require((row['source_id'], row['run_id'], row['software_reset'], row['pipe_object_id'],
        row['generation']['reset_epoch'], row['side']) ==
        (expected[0], expected[1], expected[2], expected[3], expected[5], expected[-1]), 'scope')
    require(type(row['frame_idx']) is int and row['frame_idx'] == ledger.clock + STRIDE
        and row['frame_idx'] <= ledger.deadline and row['time_sec'] == row['frame_idx']/FPS, 'clock')


def end_from_context(ledger: Any, context: dict) -> dict:
    """既存観測器の直前完了updateからだけ終了資格を作る。現在失敗票は利用しない。"""
    update, scope = context['update'], ledger.scope
    require(context['source_id'] == scope[0] and context['run_id'] == scope[1], 'context_identity')
    require(context['capture_status'] == 'CAPTURED' and not context['failures']
        and not any(context['upstream_failures'].values()) and update['returned'] is True
        and update['exception'] is None, 'context_success')
    require(context['frame_idx'] == context['available_frame'] == update['returned_frame_idx']
        == ledger.clock and context['time_sec'] == update['returned_time_sec'] == ledger.clock/FPS,
        'context_clock')
    generation = context['generation']['after']['value'][scope[-1]]
    require(generation['reset_epoch'] == scope[5] and generation['side'] == scope[-1], 'context_generation')
    require(all(update[key + '_observed'] is True and update[key] is True
        for key in ('match_end_locked', 'post_match_lockdown_active')), 'context_end_not_observed')
    return dict(source_id=scope[0], run_id=scope[1], frame=ledger.clock,
        match_end_locked=True, post_match_lockdown_active=True,
        source_capture_token=context['capture_token'], qualification='original_completed_context')


def prepare(module: Any, ledger: Any, row: dict, end: dict) -> dict:
    """原Jが保存したinactive行と同runの終了票を検査する純関数。接続側で原票を認証する。"""
    module.check(ledger)
    validate_scope(ledger, row)
    require(row['kind'] == 'enqueue' and row['status'] == 'returned'
        and row['active'] is False and row['returned_none'] is True, 'original_status')
    require(end['source_id'] == ledger.scope[0] and end['run_id'] == ledger.scope[1]
        and type(end['frame']) is int and ledger.start <= end['frame'] <= ledger.clock
        and end['match_end_locked'] is True and end['post_match_lockdown_active'] is True, 'end_evidence')
    tokens = tuple(arrival.token for arrival in ledger.arrivals)
    require(ledger.applied == tokens, 'unapplied_arrivals')
    unacked = ledger.arrivals[len(ledger.acknowledgements):]
    require(row['added_occurrence_tokens'] == [] and row['fifo_occurrence_tokens'] == []
        and row['before']['pending_tsumo'] == [] and row['after']['pending_tsumo'] == []
        and row['after']['tsumo_count'] == {}, 'inactive_fifo')
    require(row['discarded_tokens'] == [arrival.token for arrival in unacked], 'discard_tokens')
    beginning = row['update_begin_accounting']
    require(beginning is not None and beginning['pending_tsumo'] ==
        [list(arrival.pair) for arrival in unacked], 'discard_source')
    return dict(kind='arrival_scope_terminal/v1', frame=row['frame_idx'],
        source_call_token=row['token'], last_live_ledger=asdict(ledger), end_evidence=dict(end),
        unacknowledged_terminal_tokens=[arrival.token for arrival in unacked],
        original_clear_observed=bool(unacked), acknowledged_by_this_event=False,
        physical_applied_by_this_event=False, further_old_scope_updates_allowed=False,
        quality_gate_clear=False)
