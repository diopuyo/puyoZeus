"""原J・同update context・生Registryを照合し既存joint評価へ束縛する。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import belief as B
import joint_evaluation as J
import journal_context as C

REGISTRY_KEY = '_g2_probabilistic_scope_registry'
SIDES = ('1P', '2P')


@dataclass(frozen=True)
class Bound:
    values: tuple[Any, Any]
    observed: tuple[Any, Any]
    inputs: Any
    frame: int
    tokens: tuple[str, str]
    digest: str


def current(rec: Any, factory: Any, pipe: Any, journal: Any, registry: Any,
            bindings: tuple[Any, Any], contract: Any, steps: list[Any]) -> Bound:
    C.require(registry.factory is factory and getattr(factory, REGISTRY_KEY) is registry, 'registry_owner')
    C.require(rec.installed and not rec.closed and rec.active is None and not rec.errors and rec.rows, 'context_lifetime')
    C.require(journal.active is None and not journal.closed and not journal.errors, 'J_lifetime')
    row = rec.rows[-1]
    values = tuple(registry.current(binding) for binding in bindings)
    C.require(len(values) == 2, 'both_registry_bindings')
    tokens = C.join(row, steps, values[0].scope)
    for token, side in zip(tokens, SIDES, strict=True):
        ordinal = int(token.removeprefix('step:'))
        C.require(ordinal < journal.steps and journal.expected[ordinal] == (row['frame_idx'], side), 'J_issued')
    registration = dict(source_id=rec.source_id, run_id=rec.run_id,
        time_base_numerator=1, time_base_denominator=60, ledger_connection='NOT_CONNECTED')
    contract._identity(row, registration)
    contract._update(row)
    contract._generation(row)
    C.require(not row['hold_reasons'] and all(set(row['sides'][s]['hold_reasons']) <= {'next_missing'}
              for s in SIDES), 'nonlegacy_hold')
    observed = observations(pipe, row, values, steps)
    J.inputs(values, row['frame_idx'], ('STABLE', 'STABLE'), observed)
    return Bound(values, observed, contract._inputs(row), row['frame_idx'], tokens, contract.digest(row))


def observations(pipe: Any, row: Any, values: Any, steps: Any) -> tuple[Any, Any]:
    result = []
    for side, value, step in zip(SIDES, values, steps, strict=True):
        sm = getattr(pipe, '_sm_' + side.lower())
        scope = (row['source_id'], row['run_id'], step['software_reset'], id(pipe),
                 id(sm), step['generation_after']['reset_epoch'], side)
        C.require(value.scope == scope, 'live_scope')
        C.require(sm.context.frame_idx == row['frame_idx'] and sm.context.state.value == 'stable', 'live_STABLE')
        C.require(getattr(pipe, '_active_chain_' + side.lower()) is None, 'live_origin')
        grid = B.grid(sm.context.confirmed_board)
        saved = row['sides'][side]['before_hold']['confirmed']['grid']
        C.require([list(r) for r in grid] == saved, 'live_context_grid')
        result.append(B.Board.from_dict({'grid': grid}))
    return tuple(result)


def evaluate(capture: Callable[[], Bound], scorer_factory: Callable[[Any], Any], *,
             sample_count: int = J.DEFAULT_SAMPLES, seed: int = 0) -> tuple[Bound, Any]:
    before = capture()
    result = J.evaluate(before.values, before.frame, ('STABLE', 'STABLE'), before.observed,
                        scorer_factory(before.inputs), sample_count=sample_count, seed=seed)
    after = capture()
    C.require(before.digest == after.digest and before.tokens == after.tokens
              and all(a is b for a, b in zip(before.values, after.values, strict=True)), 'evaluation_state_changed')
    return before, result
