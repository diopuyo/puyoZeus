"""完全なprivate familyと原Jから起点候補を判定。Mode/台帳/公開には一切書かない。"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any
import math
import prefix_phase_math_v2 as P

FPS = 60


def source_origin(parts: Any, ledger: Any, step: dict, origin: dict) -> tuple:
    check = parts.mode.B.require
    replay = parts.arrival_saved.V.R
    replay.scope(step, ledger)
    check(step['kind'] == 'step' and step['status'] == 'returned' and step['exception'] is None,
        'identified_origin_source_call')
    check(type(origin.get('object_id')) is int and origin['object_id'] > 0,
        'identified_origin_object')
    matching = [event['active_origin'] for event in step['events'] if event.get('active_origin') is not None
        and event['active_origin']['object_id'] == origin['object_id']]
    check(bool(matching) and origin in matching, 'identified_origin_not_in_call')
    triggers = {item['trigger_sec'] for item in matching}
    check(len(triggers) == 1, 'identified_origin_trigger_ambiguous')
    trigger = next(iter(triggers))
    check(type(trigger) in (int, float) and math.isfinite(trigger)
        and ledger.start / FPS <= trigger <= step['time_sec'], 'identified_origin_trigger_clock')
    check(all(item['before_board'] is not None and item['before_board']['grid'] == origin['before_board']['grid']
        for item in matching), 'identified_origin_mutation')
    board = parts.mode.B.Board.from_dict({'grid': origin['before_board']['grid']})
    return parts.mode.B.grid(board), trigger


def family_target(parts: Any, ledger: Any, base_applied: tuple, family: Any,
                  grid: tuple, trigger: float, source_frame: int) -> tuple:
    engine, check = parts.mode.C.T, parts.mode.B.require
    check(type(family) is P.Family, 'identified_family_type')
    check(type(family.prefix) is int and family.prefix > 0 and family.phase == P.PREPOP
        and family.publication_permission is False, 'identified_family_phase')
    value = family.value
    engine.B.validate(value)
    check(value.scope == ledger.scope and value.deadline == ledger.deadline, 'identified_family_scope')
    total = len(base_applied) + family.prefix
    check(len(ledger.applied) < total <= len(ledger.arrivals), 'identified_family_prefix_range')
    expected = tuple(arrival.token for arrival in ledger.arrivals[:total])
    check(len(value.tokens) >= total and value.tokens[-total:] == expected, 'identified_family_token_prefix')
    arrival = ledger.arrivals[total - 1]
    check(arrival.frame <= value.frame <= source_frame <= value.deadline
        and value.frame / FPS <= trigger, 'identified_family_clock')
    check(all(world.grid[engine.B.HIDDEN_ROWS:] == grid[engine.B.HIDDEN_ROWS:] for world in value.worlds),
        'identified_family_visible')
    simulator = engine.B.ChainSimulator(exclude_hidden_row_from_pop=True)
    check(all(simulator.find_erasable_groups(engine.B.Board.from_dict({'grid': world.grid}))
        for world in value.worlds), 'identified_family_not_prepop')
    return total - 1, value.frame


def select(parts: Any, ledger: Any, base_applied: tuple, families: tuple,
           step: dict, origin: dict) -> dict:
    """絶対到来indexを返す非binding票。ACK済みであることは要求しない。"""
    parts.mode.L.check(ledger)
    check = parts.mode.B.require
    check(type(base_applied) is tuple and base_applied == ledger.applied[:len(base_applied)]
        and len(base_applied) <= len(ledger.applied), 'identified_base_prefix')
    check(type(families) is tuple and bool(families), 'identified_families_missing')
    grid, trigger = source_origin(parts, ledger, step, origin)
    targets = {family_target(parts, ledger, base_applied, family, grid, trigger, step['frame_idx'])
        for family in families}
    check(len(targets) == 1, 'identified_family_assignment_ambiguous')
    index, observed_frame = next(iter(targets))
    arrival = ledger.arrivals[index]
    return dict(kind='identified_origin_candidate/v1', source_call_token=step['token'],
        source_frame=step['frame_idx'], observed_frame=observed_frame, absolute_arrival_index=index,
        relative_prefix=index + 1 - len(base_applied), origin_object_id=origin['object_id'],
        origin_trigger_sec=trigger,
        arrival=asdict(arrival), base_applied_tokens=base_applied,
        head_mismatch=index != len(ledger.applied), ack_required=False,
        non_binding=True, original_origin_changed=False, original_fifo_changed=False,
        ledger_updated=False, creation_call_witnessed=False, requires_live_owner_check=True,
        stable_qualification_replayed=False, publication_permission=False, quality_gate_clear=False)
