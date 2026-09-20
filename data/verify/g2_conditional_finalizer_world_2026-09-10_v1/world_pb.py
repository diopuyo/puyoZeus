"""同call保存PBと別条件PBを照合し、原整数currentと混ぜない。"""
from __future__ import annotations
from dataclasses import fields
import hashlib
import json
import math
from typing import Any

FPS, EPSILON = 60, 1e-12
EVENT = 'conditional_current_after_original_J'
KINDS = frozenset(('conditional_hidden_current/v1', 'conditional_hidden_after_firing_current/v1',
    'conditional_hidden_after_firing_next_current/v1'))
FALSE_FIELDS = ('physical_certified', 'integer_current_permission', 'accounting_permission', 'production_permission')


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError('conditional_world:' + reason)


def equal(lib: Any, left: Any, right: Any, reason: str) -> None:
    require(lib.C.encoded(left) == lib.C.encoded(right), reason)


def probability(lib: Any, value: Any) -> None:
    require(type(value) in (list, tuple) and len(value) == lib.P.ROWS, 'PB_rows')
    for row in value:
        require(type(row) in (list, tuple) and len(row) == lib.P.COLS, 'PB_cols')
        for cell in row:
            require(type(cell) in (list, tuple) and bool(cell), 'PB_cell')
            require(all(type(pair) in (list, tuple) and len(pair) == 2 for pair in cell), 'PB_pairs')
            colors, probs = zip(*cell, strict=True)
            require(all(type(c) is int and c in lib.P.VALUES | {lib.P.UNKNOWN} for c in colors), 'PB_colors')
            require(list(colors) == sorted(set(colors)), 'PB_unique_sorted')
            require(all(type(p) is float and math.isfinite(p) and 0 <= p <= 1 for p in probs), 'PB_probabilities')
            require(abs(math.fsum(probs) - 1) <= EPSILON, 'PB_sum')


def certificate(lib: Any, event: Any) -> tuple[Any, Any]:
    value = event['current']
    require(set(value) == {field.name for field in fields(lib.C.ConditionalCurrent)}, 'certificate_fields')
    require(event['original_result_unchanged'] is True and event['integer_current_recovered'] is False, 'event_permissions')
    require(value['provisional'] is True and all(value[k] is False for k in FALSE_FIELDS), 'certificate_permissions')
    restored = lib.C.ConditionalCurrent(**{f.name: value[f.name]
        for f in fields(lib.C.ConditionalCurrent) if f.init})
    require(lib.C.intact(restored), 'original_intact')
    proof = json.loads(value['evidence_json'])
    require(proof['kind'] in KINDS and proof['physical_certified'] is False, 'current_kind')
    require(all(proof.get(key, False) is False for key in FALSE_FIELDS), 'proof_permissions')
    require(type(value['frame']) is int and value['frame'] == event['frame']
        and type(value['clock']) is float and value['clock'] == value['frame'] / FPS, 'current_clock')
    require(type(value['action']) is int and value['action'] >= 0, 'current_action')
    probability(lib, proof['original_PB']); probability(lib, proof['conditional_PB'])
    return restored, proof


def scope(lib: Any, value: Any, row: Any, step: Any, factory: Any) -> bool:
    supplied, actual = value.scope, row['decision']['history_state']['scope']
    require(type(supplied) in (list, tuple) and len(supplied) == 7, 'scope7')
    require(all(type(supplied[i]) is int for i in (2, 3, 4, 5)), 'scope_types')
    equal(lib, [supplied[0], supplied[1], supplied[2], supplied[-1]],
        ['sha256:' + actual['source_sha256'], actual['run_id'], actual['reset_epoch'], actual['side']], 'owner_scope')
    equal(lib, [supplied[0], supplied[1], supplied[2], supplied[3], supplied[5], supplied[-1]],
        [step['source_id'], step['run_id'], step['software_reset'], step['pipe_object_id'],
         step['generation']['reset_epoch'], step['side']], 'J_scope')
    for name in ('source_id', 'run_id', 'pipe_object_id', 'side', 'generation'):
        equal(lib, row['scope'][name], step[name], 'history_J_scope:' + name)
    if factory is None:
        return False
    pipe = factory.provider.journal.tracker._pipeline
    equal(lib, supplied, lib.E.scope(factory, pipe), 'actual_live_scope7')
    equal(lib, supplied, factory.controller.history[supplied[-1]].scope, 'live_binding_scope')
    return True


def grids(lib: Any, value: Any, proof: Any, row: Any) -> None:
    sm, world = lib.P.grid(value.sm_grid), lib.P.grid(value.inferred_grid)
    raw = lib.P.grid(proof['raw'], hidden_unknown=True)
    hidden = lib.P.HIDDEN_ROWS
    require(raw[hidden:] == sm[hidden:] == world[hidden:] and lib.P.compatible(raw, world), 'raw_world_visible')
    equal(lib, proof['raw_capture']['raw_grid'], raw, 'capture_raw')
    capture = proof['raw_capture']['capture']
    require(proof['raw_capture']['captured_frame'] == capture['captured_frame'] == value.frame, 'capture_clock')
    equal(lib, capture, row['raw_capture'], 'same_call_raw_capture')
    for name in ('raw', 'filtered'):
        equal(lib, capture[name]['grid'], raw, 'original_capture_grid')
    equal(lib, row['grid_after'], world, 'private_world')
    equal(lib, value.integer_anchor, lib.C.encoded(row['decision']['history_state']['current']), 'old_slot')
    require(row['decision']['current_permission'] is False and row['decision'].get('current_proof') is None, 'integer_not_published')
    require(value.action == row['decision']['history_state']['action'], 'same_action')
    require(row['window'] is False and not row['view_tokens'] and row['next_token'] is None, 'current_pending_or_window')
    digest = hashlib.sha256(bytes(c for r in sm[hidden:] for c in r)).hexdigest()
    require(proof['visible_sha256'] == digest, 'visible_digest')
    for r in range(lib.P.ROWS):
        for c in range(lib.P.COLS):
            equal(lib, proof['conditional_PB'][r][c], [[world[r][c], 1.0]], 'conditional_cell')
            if r >= lib.P.HIDDEN_ROWS:
                equal(lib, proof['original_PB'][r][c], proof['conditional_PB'][r][c], 'visible_PB_unchanged')


def original(lib: Any, value: Any, proof: Any, row: Any, step: Any, outer: Any) -> None:
    require(outer['frame_idx'] == value.frame and outer['time_sec'] == value.clock, 'outer_clock')
    require(outer['same_result_identity'] is True and not outer['changed_sides'], 'outer_identity')
    equal(lib, outer['full_before'], outer['full_after'], 'original_typed_unchanged')
    typed = lib.typed_fields(outer['full_after'][value.scope[-1]])
    equal(lib, typed['state'], ['src.board_state_machine.BoardState', ['builtins.str', 'stable']], 'natural_STABLE')
    equal(lib, lib.typed_pb(typed['prob_board']), proof['original_PB'], 'actual_original_PB')
    digest = hashlib.sha256(bytes(c for r in value.sm_grid for c in r)).hexdigest()
    equal(lib, lib.typed_fields(typed['confirmed_board'])['_grid'],
        ['numpy.ndarray', '|u1', [lib.P.ROWS, lib.P.COLS], digest], 'actual_confirmed')
    require(step['kind'] == 'step' and step['status'] == 'returned' and step['exception'] is None, 'J_return')
    require(step['returned']['state'] == 'STABLE' and row['original_state'] == 'stable', 'J_natural_STABLE')
    require(step['token'] == row['journal_token'] and step['code_sha256'] == row['code_sha256'], 'same_J_call')
    require(step['frame_idx'] == row['scope']['frame_idx'] == value.frame, 'same_J_frame')
    require(step['time_sec'] == row['scope']['time_sec'] == value.clock, 'same_J_clock')
    equal(lib, step['returned']['confirmed']['grid'], value.sm_grid, 'J_confirmed')
    equal(lib, row['returned']['grid'], value.sm_grid, 'history_confirmed')
    transition = proof.get('natural_transition')
    if transition is not None:
        require(transition['frame'] == value.frame and transition['natural_choice'] is True
            and transition['raw_unchanged'] is True, 'natural_transition')
        equal(lib, transition['actual_current'], value.sm_grid, 'natural_transition_current')
