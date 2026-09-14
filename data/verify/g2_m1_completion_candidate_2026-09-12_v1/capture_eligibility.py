"""正常な未資格だけをHOLDへ分類し、原票不整合は例外で止める。"""
from __future__ import annotations

import math
from typing import Any
import evaluation_flags as F

SIDES = ('1P', '2P')
NORMAL_SIDE_HOLDS = frozenset(('next_missing', 'non_stable', 'before_probability_invalid_or_missing'))
NORMAL_COMMON_HOLDS = frozenset(('is_match_active_not_ready', 'match_end_locked_not_ready',
                                'post_match_lockdown_active_not_ready'))


def side_flags(flag: dict, step: dict, frame: int) -> list[str]:
    F.require(flag['token'] == step['token'] and flag['software_reset'] == step['software_reset'], 'issued_flag')
    for key in ('source_id', 'run_id', 'frame_idx', 'time_sec', 'side', 'pipe_object_id', 'generation'):
        F.require(flag['scope'][key] == step[key], 'flag_scope')
    F.require(step['frame_idx'] == frame and step['status'] == 'returned' and step['exception'] is None, 'step_return')
    side = step['side']
    if flag['hold_reason'] is not None: return [side + ':' + flag['hold_reason']]
    reasons = []
    for key, expected in (('match_active', True), ('effect_window', False), ('origin_present', False)):
        F.require(type(flag[key]) is bool, 'flag_type')
        if flag[key] is not expected: reasons.append(side + ':' + key)
    if flag['state'] != 'stable': reasons.append(side + ':not_stable')
    end = flag['grace_end']
    F.require(end is None or type(end) in (int, float) and math.isfinite(end), 'grace_type')
    if end is not None and step['time_sec'] < end: reasons.append(side + ':landing_grace')
    if step['generation'] != step['generation_after']: reasons.append(side + ':in_call_generation_change')
    return reasons


def reasons(session: Any, frame: int) -> tuple[str, ...]:
    flags, witness, row = session.evaluation_flags, session.witness, session.state['provisional_context_observer'].rows[-1]
    F.require(not flags.closed and flags.error is None and flags.journal is witness.journal, 'flags_lifetime')
    F.require(row['capture_status'] == 'CAPTURED' and not row['failures']
              and not any(row['upstream_failures'].values()), 'context_failure')
    steps = witness.pair(frame)
    F.require([s['side'] for s in steps] == list(SIDES) and set(flags.latest) == set(SIDES), 'both_sides')
    found = []
    for side, step in zip(SIDES, steps):
        found.extend(side_flags(flags.latest[side], step, frame))
        holds = set(row['sides'][side]['hold_reasons'])
        F.require(holds <= NORMAL_SIDE_HOLDS, 'unexpected_context_hold')
        probability = row['sides'][side]['before_hold']['probability']
        F.require(probability['present'] is False or probability['type_valid'] is True
                  and not probability['errors'], 'invalid_present_probability')
        found.extend(side + ':' + h for h in sorted(holds - {'next_missing'}))
    F.require(set(row['hold_reasons']) <= NORMAL_COMMON_HOLDS, 'unexpected_common_hold')
    found.extend(row['hold_reasons'])
    modes = (session.state['probabilistic_tracking_mode'], session.mode)
    for side, mode in zip(SIDES, modes):
        F.require(mode is None or mode.error is None, 'mode_failure')
        if mode is None or getattr(mode, 'closed', False) or mode.connection.binding is None:
            found.append(side + ':basis_missing')
            continue
        F.require(mode.error is None and mode.native is not None, 'mode_failure')
        if mode.native.pending: found.append(side + ':native_pending')
        ledger = getattr(mode, 'arrival_ledger', None)
        if side == '1P': F.require(ledger is not None, 'first_arrival_ledger_missing')
        if ledger is not None and (len(ledger.applied) != len(ledger.arrivals)
                                  or len(ledger.acknowledgements) != len(ledger.arrivals)):
            found.append(side + ':arrival_pending')
    return tuple(dict.fromkeys(found))
