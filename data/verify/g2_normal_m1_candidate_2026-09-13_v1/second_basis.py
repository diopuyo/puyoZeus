"""非reset側の有資格STABLE観測から初回だけ確率基準を登録する。"""
from __future__ import annotations

from typing import Any
import hashlib
import inspect
from pathlib import Path
import belief as B
import conditioning as P
import serialization as S
import journal_context as C

SIDE = '2P'
POLICY_SOURCE = Path(__file__).resolve().parent.parent/'g2_hidden_basis_initialization_2026-09-11_v1/hidden_basis_gate.py'
POLICY_SHA = '46b6db562f04dc8e41a5654bd02e38c67f359a70d35ad6412db59a811e7aaf83'


def policy_guard(policy: Any) -> None:
    data = POLICY_SOURCE.read_bytes()
    C.require(hashlib.sha256(data).hexdigest() == POLICY_SHA, 'second_policy_source')
    code = next(value for value in compile(data, str(POLICY_SOURCE), 'exec').co_consts
                if inspect.iscode(value) and value.co_name == 'initial_hidden')
    C.require(policy.initial_hidden.__code__ == code and policy.initial_hidden.__globals__ is vars(policy), 'second_policy_function')
    C.require(policy.PRIOR_POLICY == 'native_distribution_or_uniform_for_raw_unknown/v1'
        and policy.EMPTY_COLOR == 0 and policy.G.COLOR_UNKNOWN == B.COLOR_UNKNOWN
        and policy.UNOBSERVED_PRIOR == tuple((color,1.0/len(B.PROB_COLORS)) for color in B.PROB_COLORS), 'second_policy_constants')


class BasisHold(ValueError):
    """正常な未資格観測。基準を作らず次の観測へ進む。"""


def qualify(evidence: Any, witness: Any, pipe: Any, side: str = SIDE) -> tuple[Any, Any]:
    C.require(side in ('1P', '2P'), 'normal_basis_side')
    C.require(not evidence.closed and evidence.error is None and evidence.latest is not None, 'second_observation_live')
    C.require(evidence.journal is witness.journal and witness.journal.pipe is pipe, 'second_observer_owner')
    row = evidence.latest
    if row.get('hold_reason') or row.get('state') != 'stable':
        raise BasisHold('second_not_STABLE')
    if not (row['match_active'] is True and row['effect_window'] is False
            and row['origin_present'] is False and row['landing_grace_expired'] is True):
        raise BasisHold('second_not_settled')
    C.require(row['pb_holds'] == row['pb_errors'] == [], 'second_probability_errors')
    step = witness.pair(row['frame'])[0 if side == '1P' else 1]
    C.require(step['side'] == side and step['status'] == 'returned' and step['exception'] is None, 'second_completed_J')
    sm = getattr(pipe, '_sm_' + side.lower())
    scope = [step['source_id'], step['run_id'], step['software_reset'], id(pipe), id(sm),
             step['generation_after']['reset_epoch'], side]
    C.require(row['scope'] == scope and row['journal_token'] == step['token']
        and row['original_code_sha256'] == step['code_sha256'], 'second_basis_J')
    C.require(step['generation'] == step['generation_after'] and step['pipe_object_id'] == id(pipe), 'second_basis_generation')
    C.require(sm.context.frame_idx == row['frame'] and sm.context.state.value == 'stable'
        and getattr(pipe, '_active_chain_' + side.lower()) is None, 'second_basis_current')
    grid = B.grid(sm.context.confirmed_board)
    C.require([list(r) for r in grid] == row['confirmed']['grid'] == step['returned']['confirmed']['grid'], 'second_basis_grid')
    return row, grid


def distribution(row: Any, grid: Any, policy: Any) -> tuple[Any, Any]:
    policy_guard(policy)
    raw = tuple(map(tuple, row['raw']['grid']))
    C.require(raw[B.HIDDEN_ROWS:] == grid[B.HIDDEN_ROWS:], 'second_raw_visible')
    saved = row['probability']
    C.require(saved['present'] is True and saved['type_valid'] is True and saved['errors'] == [], 'second_PB_valid')
    board = B.Board.from_dict({'grid': grid})
    probability = B.ProbabilisticBoard.from_board(board)
    C.require(len(saved['cells']) == B.BOARD_ROWS, 'second_PB_rows')
    for r, cells in enumerate(saved['cells']):
        C.require(len(cells) == B.BOARD_COLS, 'second_PB_columns')
        for c, pairs in enumerate(cells):
            C.require(len(dict(pairs)) == len(pairs), 'second_duplicate_probability')
            probability.cell(r,c).probs = dict(pairs)
            B.distribution(probability.cell(r,c))
    hidden = tuple(B.distribution(probability.cell(0,c)) for c in range(B.BOARD_COLS))
    derived, changed, reason = policy.initial_hidden(raw, grid, hidden)
    C.require(reason is None, 'second_hidden_prior:' + str(reason))
    for c, values in enumerate(derived):
        probability.cell(0,c).probs = dict(values)
    return probability, changed


def initialize(evidence: Any, witness: Any, pipe: Any, registry: Any, factory: Any,
               deadline: int, policy: Any, side: str = SIDE) -> tuple[Any, dict[str, Any]]:
    C.require(registry.factory is factory and getattr(factory, '_g2_probabilistic_scope_registry') is registry, 'second_registry_owner')
    row, grid = qualify(evidence, witness, pipe, side=side)
    probability, changed = distribution(row, grid, policy)
    value, mass, removed = P.establish_conditioned(tuple(row['scope']), row['frame'], deadline,
        B.Board.from_dict({'grid': grid}), probability)
    receipt = dict(state=S.encode(value), source_call_token=row['journal_token'],
        initial_observation=row, retained_mass=mass, gravity_removed_worlds=removed,
        newly_unobserved_columns=list(changed), hidden_prior_policy=policy.PRIOR_POLICY,
        hidden_prior_calibrated=False, reset_or_fall_claimed=False, quality_gate_clear=False)
    binding = registry.bind(factory, value, row['journal_token'])
    return binding, receipt
