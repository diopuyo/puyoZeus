"""単一PREPOPの診断入力を構築する。確定盤面/会計/勝率へは供給しない。"""
from dataclasses import asdict
import hashlib
from typing import Any

FPS = 60
SIDE = '1P'


def packet(frame: int, reason: str, prediction: Any = None) -> dict:
    return dict(kind='conditional_projected_input/v1', status='HOLD' if prediction is None else 'READY',
        cutoff_frame=frame, side=SIDE, reason=reason,
        prediction=None if prediction is None else asdict(prediction),
        accounting_permission=False, future_fire_power_supply_authorized=False,
        source_producer_authorized=False, live_hook_verified=False, quality_gate_clear=False,
        calibrated_win_probability=False)


def read(session: Any, lease: Any, reader: Any, source: Any, projection: Any, frame: int, *,
         settled: Any) -> dict:
    lane, step, reason = source.read(session, lease, reader, frame)
    if reason is not None:
        return packet(frame, reason)
    origin = (step.get('returned') or {}).get('active_origin')
    if origin is None:
        return packet(frame, 'returned_origin_not_available')
    if settled(origin):
        return packet(frame, 'returned_origin_is_settled_notice')
    if origin.get('before_board') is None:
        return packet(frame, 'returned_origin_board_not_available')
    matching = [event['active_origin'] for event in step['events']
                if event.get('active_origin') is not None
                and event['active_origin']['object_id'] == origin['object_id']]
    if not matching:
        return packet(frame, 'returned_origin_not_observed_in_call')
    if any(value['trigger_sec'] != origin['trigger_sec'] or value['before_board'] != origin['before_board']
           for value in matching):
        raise ValueError('projected_input_origin_mutated')
    saved = origin['before_board']
    if hashlib.sha256(projection.encoded(saved['grid']).encode()).hexdigest() != saved['sha256']:
        raise ValueError('projected_input_source_board_hash')
    live, _ = lease.current()
    family = lane.families[0]
    grid = live.module.B.grid(live.module.B.Board.from_dict(origin['before_board']))
    hidden = live.module.B.HIDDEN_ROWS
    if family.value.frame / FPS > origin['trigger_sec'] or any(
            world.grid[hidden:] != grid[hidden:] for world in family.value.worlds):
        return packet(frame, 'returned_origin_not_represented_by_family')
    mode = session.state[source.MODE_KEY]
    result = projection.project(live.parts, mode.arrival_ledger, lane, step, origin)
    return packet(frame, 'conditional_single_prepop', result)
