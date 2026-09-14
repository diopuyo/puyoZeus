"""原Jのoriginと現在STABLE可視を使い、保存posteriorを原数学で再計算する。"""
from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any

import base_mode as BASE
import cascade_arrival as C
import ledger as L
import source_replay as R

B, S = BASE.B, BASE.V1.S


def origin(saved: dict, steps: dict[str, dict], ledger: L.Ledger, frame: int) -> Any:
    step = steps[saved['source_call_token']]
    R.scope(step, L.Ledger(ledger.scope, ledger.start, ledger.deadline, ledger.start))
    L.require(step['frame_idx'] == saved['first_observed_frame'] <= frame
              and step['status'] == 'returned' and step['exception'] is None, 'replay_origin_call')
    matching = [e['active_origin'] for e in step['events'] if e.get('active_origin') is not None
                and e['active_origin']['object_id'] == saved['object_id']]
    L.require(bool(matching) and all(o['before_board'] is not None
        and R.normalized(o['before_board']['grid']) == R.normalized(saved['grid']) for o in matching), 'replay_origin_grid')
    triggers = {o['trigger_sec'] for o in matching}
    L.require(len(triggers) == 1 and saved['creation_call_witnessed'] is False
        and all(o['chain_count'] == saved['estimated_chain_count'] for o in matching), 'replay_origin_ambiguous')
    trigger = next(iter(triggers))
    L.require(type(trigger) in (int, float) and math.isfinite(trigger)
              and ledger.start/R.FPS <= trigger <= step['time_sec'], 'replay_origin_clock')
    return B.Board.from_dict({'grid': saved['grid']})


def basis_origin(saved: dict, steps: dict, ledger: L.Ledger, frame: int, initial_call: str) -> Any:
    L.require(bool(initial_call) and saved['operation_token'] ==
              'basis-cascade:' + initial_call + ':' + saved['source_call_token'], 'replay_basis_token')
    return origin(saved, steps, ledger, frame)


def hand_origin(saved: dict, steps: dict, ledger: L.Ledger, frame: int, arrival: L.Arrival) -> Any:
    board = origin(saved, steps, ledger, frame)
    step = steps[saved['source_call_token']]
    trigger = next(e['active_origin']['trigger_sec'] for e in step['events']
        if e.get('active_origin') is not None and e['active_origin']['object_id'] == saved['object_id'])
    eligible = [a for a in ledger.arrivals[len(ledger.applied):] if a.frame / R.FPS <= trigger]
    L.require(eligible == [arrival], 'replay_origin_assignment')
    return board


def transition(current: Any, ledger: L.Ledger, row: dict, step: dict,
               steps: dict[str, dict], prior: Any, basis_closed: bool,
               initial_call: str) -> tuple[Any, L.Ledger, bool]:
    receipt = row['transition']
    frame, kind = step['frame_idx'], receipt['kind']
    L.require(receipt['source_call_token'] == step['token'] and receipt['applied_frame'] == frame
              and receipt['physical_certified'] is False, 'replay_transition_call')
    L.require(step['returned'] is not None and step['returned']['state'] == 'STABLE', 'replay_transition_STABLE')
    observed = B.Board.from_dict({'grid': step['returned']['confirmed']['grid']})
    if kind == 'basis_cascade':
        L.require(not basis_closed and not ledger.applied, 'replay_duplicate_basis')
        before = basis_origin(receipt['origin'], steps, ledger, frame, initial_call)
        following, report = BASE.V1.C.settle(current, current.scope, frame,
            receipt['origin']['operation_token'], before, observed)
        basis_closed = True
    else:
        following, report, ledger, basis_closed = arrived(current, ledger, receipt, observed, steps, prior, basis_closed, initial_call)
    L.require(S.encode(following) == receipt['state']
        and R.normalized(asdict(report)) == receipt['distribution_report'], 'replay_posterior_or_report')
    return following, ledger, basis_closed


def arrived(current: Any, ledger: L.Ledger, receipt: dict, observed: Any,
            steps: dict[str, dict], prior: Any, basis_closed: bool, initial_call: str) -> tuple:
    frame, kind = receipt['applied_frame'], receipt['kind']
    L.require(len(ledger.applied) < len(ledger.arrivals), 'replay_arrival_missing')
    arrival = ledger.arrivals[len(ledger.applied)]
    L.require(receipt['arrival'] == R.normalized(asdict(arrival))
              and receipt['original_fifo_changed'] is False, 'replay_arrival_identity')
    if kind == 'basis_cascade_with_arrival':
        L.require(not basis_closed and receipt['origin'] is None, 'replay_composite_origin')
        before = basis_origin(receipt['basis_origin'], steps, ledger, frame, initial_call)
        following, report = C.run(current, ledger, frame, arrival, before, observed, prior)
        basis_closed = True
    else:
        L.require(kind == 'arrived_hand' and receipt['basis_origin'] is None, 'replay_hand_kind')
        before = None if receipt['origin'] is None else hand_origin(receipt['origin'], steps, ledger, frame, arrival)
        following, report = C.T.run(current, current.scope, frame, arrival.token, arrival.pair,
            observed, 0 if before is None else None, prior, origin_observed=before)
    return following, report, L.applied(ledger, (arrival.token,), frame), basis_closed
