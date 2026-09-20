"""別の原J/履歴から候補対象を列挙し、保存候補の欠落を見逃さない。"""
from __future__ import annotations
import json
import hashlib
from typing import Any
import world_pb as B
import world_geometry as G


def eligible(lib: Any, row: Any, step: Any) -> bool:
    state = row['decision']['history_state']
    if (step['returned']['state'] != 'STABLE' or row['window'] or row['view_tokens']
            or row['next_token'] is not None or state['debts']):
        return False
    raw = lib.P.grid(row['raw_capture']['raw']['grid'], hidden_unknown=True)
    world = lib.P.grid(row['grid_after'])
    sm = lib.P.grid(step['returned']['confirmed']['grid'])
    return (raw[lib.P.HIDDEN_ROWS:] == sm[lib.P.HIDDEN_ROWS:] == world[lib.P.HIDDEN_ROWS:]
        and lib.P.compatible(raw, world))


def original_state(lib: Any, row: Any, step: Any, outer: Any) -> None:
    side = row['scope']['side']
    B.require(outer['same_result_identity'] is True and not outer['changed_sides'], 'coverage_outer_identity')
    B.equal(lib, outer['full_before'], outer['full_after'], 'coverage_outer_unchanged')
    typed = lib.typed_fields(outer['full_after'][side])
    state = step['returned']['state'].lower()
    B.equal(lib, typed['state'], ['src.board_state_machine.BoardState', ['builtins.str', state]],
        'coverage_actual_state')
    B.require(row['original_state'] == state, 'coverage_history_state')
    grid = lib.P.grid(step['returned']['confirmed']['grid'])
    digest = hashlib.sha256(bytes(c for cells in grid for c in cells)).hexdigest()
    B.equal(lib, lib.typed_fields(typed['confirmed_board'])['_grid'],
        ['numpy.ndarray', '|u1', [lib.P.ROWS, lib.P.COLS], digest], 'coverage_actual_confirmed')
    B.equal(lib, row['returned']['grid'], grid, 'coverage_history_confirmed')


def verify(lib: Any, histories: Any, journal: Any, currents: Any, prepared: Any, outer: Any,
           expected_current_keys: Any = None) -> None:
    steps = {(row['frame_idx'], row['side']): row for row in journal if row['kind'] == 'step'}
    active, expected, previous_state, scopes = {}, set(), {}, {}
    for key, row in histories.items():
        frame, side = key
        step = steps[key]
        original_state(lib, row, step, outer[frame])
        if key in prepared:
            active[side] = prepared[key]['kind']
        if expected_current_keys is None and active.get(side) not in (None, G.PREFIX) and eligible(lib, row, step):
            expected.add(key)
        if key in currents:
            value = currents[key]
            if side in scopes:
                B.equal(lib, value['scope'], scopes[side], 'same_segment_scope7')
            scopes[side] = value['scope']
            if previous_state.get(side) != 'STABLE':
                B.require(json.loads(value['evidence_json']).get('natural_transition') is not None,
                    'missing_natural_transition')
        previous_state[side] = step['returned']['state']
    if expected_current_keys is not None:
        expected = set(expected_current_keys)
    B.require(set(currents) == expected, 'candidate_coverage')
