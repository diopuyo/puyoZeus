"""原到来票と検証済み原popを別順序で受ける私有状態。FIFOは保持する。"""
from __future__ import annotations
from typing import Any

FPS, STRIDE = 60, 2


def source_scope(module: Any, ledger: Any, row: dict) -> None:
    scope = ledger.scope
    module.require(row['source_id'] == scope[0] and row['run_id'] == scope[1]
        and row['software_reset'] == scope[2] and row['pipe_object_id'] == scope[3]
        and row['generation']['reset_epoch'] == scope[5] and row['side'] == scope[-1],
        'second_source_scope')
    module.require(row['kind'] == 'enqueue' and row['status'] == 'returned'
        and row['active'] is True and row['time_sec'] == row['frame_idx'] / FPS,
        'second_source_status_clock')


def arrive(module: Any, ledger: Any, row: dict) -> Any:
    source_scope(module, ledger, row)
    added = row['added_occurrence_tokens']
    module.require(type(added) is list and len(added) <= 1, 'second_multiple_arrivals')
    if added:
        value = module.Arrival(ledger.scope, added[0], tuple(row['after']['pending_tsumo'][-1]),
                               row['frame_idx'], row['token'])
        ledger = module.arrive(ledger, value)
    unacked = ledger.arrivals[len(ledger.acknowledgements):]
    module.require(tuple(row['fifo_occurrence_tokens']) == tuple(a.token for a in unacked)
        and row['after']['pending_tsumo'] == [list(a.pair) for a in unacked],
        'second_source_fifo_prefix')
    return ledger


def advance(module: Any, ledger: Any, enqueues: tuple[dict, ...],
            verified_consumptions: tuple[Any, ...], cutoff: int) -> Any:
    """元Nativeで認証された消費だけ渡す。元emitだけではACK許可しない。"""
    module.check(ledger)
    module.require(type(cutoff) is int and ledger.clock < cutoff <= ledger.deadline,
                   'second_advance_cutoff')
    frames = tuple(row['frame_idx'] for row in enqueues)
    module.require(frames == tuple(range(ledger.clock + STRIDE, cutoff + STRIDE, STRIDE)),
                   'second_enqueue_coverage')
    ack_frames = tuple(event.consumed_frame for event in verified_consumptions)
    module.require(len(set(ack_frames)) == len(ack_frames)
                   and all(frame in frames for frame in ack_frames), 'second_ack_clock')
    events = {event.consumed_frame: event for event in verified_consumptions}
    following = ledger
    for row in enqueues:
        following = arrive(module, following, row)
        event = events.get(row['frame_idx'])
        if event is not None:
            following = module.acknowledge(following, event.occurrence_token, event.pair,
                                          event.consumed_frame, event.source_call_token)
    return module.advance_clock(following, cutoff)
