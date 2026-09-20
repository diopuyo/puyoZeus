"""同run原J/context/1P台帳/2P消費再構成から採録理由を独立に導く。"""
from __future__ import annotations

import math

SIDES = ('1P', '2P')
SCOPE_KEYS = ('source_id', 'run_id', 'frame_idx', 'time_sec', 'side', 'pipe_object_id', 'generation')


def require(ok: bool, reason: str) -> None:
    if not ok: raise ValueError('qualification_saved:' + reason)


def flags(flag: dict, step: dict) -> list[str]:
    require(all(flag['scope'][k] == step[k] for k in SCOPE_KEYS)
            and flag['token'] == step['token'] and flag['software_reset'] == step['software_reset'], 'flag_J')
    require(step['status'] == 'returned' and step['exception'] is None, 'J_failure')
    side = step['side']
    if flag['hold_reason'] is not None:
        require(flag['hold_reason'] == 'signals_or_result_missing', 'unknown_flag_hold')
        return [side + ':' + flag['hold_reason']]
    require(step['returned']['state'].lower() == flag['state']
            and (step['returned']['active_origin'] is not None) is flag['origin_present'], 'flag_returned')
    reasons = []
    for key, wanted in (('match_active', True), ('effect_window', False), ('origin_present', False)):
        require(type(flag[key]) is bool, 'flag_type')
        if flag[key] is not wanted: reasons.append(side + ':' + key)
    if flag['state'] != 'stable': reasons.append(side + ':not_stable')
    grace = flag['grace_end']
    require(grace is None or type(grace) in (int, float) and math.isfinite(grace), 'grace')
    if grace is not None and step['time_sec'] < grace: reasons.append(side + ':landing_grace')
    if step['generation'] != step['generation_after']: reasons.append(side + ':in_call_generation_change')
    return reasons


def side_context(data: dict, side: str) -> list[str]:
    holds = [key + '_missing' for key in ('pb', 'sm', 'next', 'candidate_row') if data[key] is None]
    if data['before_hold']['state'] != 'STABLE': holds.append('non_stable')
    holds.extend(k for k, v in data['identity'].items() if v is False and k != 'before_pb_is_after_pb')
    probability = data['before_hold']['probability']
    if probability['errors']: holds.append('before_probability_invalid_or_missing')
    require(data['hold_reasons'] == holds, 'context_side_hold')
    require(set(holds) <= {'next_missing', 'non_stable', 'before_probability_invalid_or_missing'}, 'invalid_side_hold')
    require(probability['present'] is False or probability['type_valid'] is True and not probability['errors'], 'probability')
    return [side + ':' + value for value in sorted(set(holds) - {'next_missing'})]


def arrival_pending(ledger: dict) -> bool:
    tokens = [a['token'] for a in ledger['arrivals']]
    applied, acknowledgements = ledger['applied'], [a['token'] for a in ledger['acknowledgements']]
    require(len(tokens) == len(set(tokens)) and applied == tokens[:len(applied)]
            and acknowledgements == tokens[:len(acknowledgements)], 'arrival_prefix')
    return len(applied) != len(tokens) or len(acknowledgements) != len(tokens)


def reasons(row: dict, captured: dict, steps: tuple[dict, dict], first: dict,
            second_pending: tuple[str, ...] | None) -> tuple[str, ...]:
    require(row['capture_status'] == 'CAPTURED' and not row['failures']
            and not any(row['upstream_failures'].values()), 'context_failure')
    require([s['side'] for s in steps] == list(SIDES) and set(captured) == set(SIDES), 'sides')
    found = []
    for side, step in zip(SIDES, steps, strict=True):
        require(all(row[k] == step[k] for k in ('source_id', 'run_id', 'frame_idx', 'time_sec')), 'context_J')
        found.extend(flags(captured[side], step))
        found.extend(side_context(row['sides'][side], side))
    update = row['update']
    common = [k + '_not_ready' for k, wanted in (('is_match_active', True), ('match_end_locked', False),
              ('post_match_lockdown_active', False)) if update.get(k + '_observed') is not True or update.get(k) is not wanted]
    require(common == row['hold_reasons'], 'context_common_hold')
    found.extend(common)
    require(all(first['scope'][k] == steps[0][k] for k in SCOPE_KEYS)
            and first['journal_token'] == steps[0]['token'], 'first_tracking_J')
    if first['pending_occurrences']: found.append('1P:native_pending')
    ledger = first['arrival_ledger']
    if arrival_pending(ledger):
        found.append('1P:arrival_pending')
    if second_pending is None: found.append('2P:basis_missing')
    elif second_pending: found.append('2P:native_pending')
    return tuple(dict.fromkeys(found))
