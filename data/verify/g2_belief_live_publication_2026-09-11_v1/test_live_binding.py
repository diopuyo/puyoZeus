"""原保存context＋実Registry、人工生pipe/2P基準で接続拒否境界を検査する。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as N
from typing import Any
import numpy as np
import pytest
import check_saved_inputs as S
import live_binding as L
import registry as R


@pytest.fixture(scope='module')
def saved() -> tuple[Any, ...]:
    value = S.S.decode([r for r in S.rows('PROBABILISTIC_TRACKING.jsonl')
                       if r['physical_transition_applied']][-1]['transition']['state'])
    row = next(r for r in S.rows('provisional_context.jsonl') if r['frame_idx'] == value.frame)
    all_steps = [r for r in S.rows('atomic_journal.jsonl') if r['kind'] == 'step']
    return value, row, all_steps, S.load_binding()


@pytest.fixture
def live(saved: tuple[Any, ...]) -> Any:
    old, source, all_steps, contract = saved
    row = deepcopy(source)
    pipe, factory = N(), N()
    for side in L.SIDES:
        board = L.B.Board.from_dict({'grid': row['sides'][side]['before_hold']['confirmed']['grid']})
        setattr(pipe, '_sm_' + side.lower(), N(context=N(frame_idx=row['frame_idx'], state=N(value='stable'), confirmed_board=board)))
        setattr(pipe, '_active_chain_' + side.lower(), None)
    steps = deepcopy([r for r in all_steps if r['frame_idx'] == row['frame_idx']])
    values = []
    for side, step in zip(L.SIDES, steps, strict=True):
        step['pipe_object_id'] = id(pipe)
        sm = getattr(pipe, '_sm_' + side.lower())
        scope = (*old.scope[:2], step['software_reset'], id(pipe), id(sm), step['generation']['reset_epoch'], side)
        # 2Pはこの単体fixtureだけの人工初期化。実producer資格の試験ではない。
        value = replace(old, scope=scope) if side == '1P' else L.B.establish(scope, old.frame, old.deadline,
            sm.context.confirmed_board, L.B.ProbabilisticBoard.from_board(sm.context.confirmed_board))
        values.append(value)
    registry = R.Registry(factory)
    setattr(factory, L.REGISTRY_KEY, registry)
    bindings = tuple(registry.bind(factory, v, 'fixture') for v in values)
    rec = N(installed=True, closed=False, active=None, errors=[], rows=[row], source_id=old.scope[0], run_id=old.scope[1])
    journal = N(active=None, closed=False, errors=[], steps=600, expected=[(r['frame_idx'], r['side']) for r in all_steps])
    capture = lambda: L.current(rec, factory, pipe, journal, registry, bindings, contract, steps)
    return N(capture=capture, pipe=pipe, rec=rec, journal=journal, registry=registry, factory=factory, bindings=bindings, steps=steps)


def test_joint_callback(live: Any) -> None:
    bound, result = L.evaluate(live.capture, lambda inputs: lambda batch: np.full(len(batch), 0.6))
    assert result.probability_p1 == pytest.approx(0.6) and len(bound.values[0].worlds) == 49
    assert bound.tokens == ('step:598', 'step:599') and not result.quality_gate_clear


@pytest.mark.parametrize('case', ['missing_J', 'active_J', 'origin', 'scope', 'old_frame', 'held', 'registry'])
def test_reject_before_model(live: Any, case: str) -> None:
    if case == 'missing_J': live.journal.steps = 599
    elif case == 'active_J': live.journal.active = object()
    elif case == 'origin': live.pipe._active_chain_2p = object()
    elif case == 'scope': live.steps[1]['software_reset'] += 1
    elif case == 'old_frame': live.pipe._sm_2p.context.frame_idx -= 2
    elif case == 'held': live.rec.rows[-1]['sides']['2P']['hold_reasons'].append('non_stable')
    elif case == 'registry': setattr(live.factory, L.REGISTRY_KEY, object())
    with pytest.raises(ValueError):
        L.evaluate(live.capture, lambda inputs: pytest.fail('invalid input reached model'))


def test_mutation_during_evaluation(live: Any) -> None:
    def scorer(batch: np.ndarray) -> np.ndarray:
        live.registry.retire(live.factory, live.bindings[1])
        return np.zeros(len(batch))
    with pytest.raises(ValueError):
        L.evaluate(live.capture, lambda inputs: scorer)
