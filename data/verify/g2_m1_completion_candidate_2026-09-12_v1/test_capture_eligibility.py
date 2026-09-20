"""原票破損と正常HOLDの分類を人工票で検査する。物理精度の証明ではない。"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace as N
from typing import Any
import pytest
import capture_eligibility as E

FRAME = 35370


def session() -> Any:
    flags, steps, sides = {}, [], {}
    journal = object()
    for side in E.SIDES:
        scope = dict(source_id='source', run_id='run', frame_idx=FRAME, time_sec=FRAME/60,
                     side=side, pipe_object_id=1, generation=dict(side=side, reset_epoch=3))
        step = dict(scope, token=side + ':1', software_reset=3, status='returned', exception=None,
                    generation_after=deepcopy(scope['generation']))
        steps.append(step)
        flags[side] = dict(scope=deepcopy(scope), token=step['token'], software_reset=3,
                          hold_reason=None, state='stable', match_active=True, effect_window=False,
                          origin_present=False, grace_end=None)
        sides[side] = dict(hold_reasons=['next_missing'],
                          before_hold=dict(probability=dict(present=True, type_valid=True, errors=[])))
    row = dict(capture_status='CAPTURED', failures=[], upstream_failures={}, sides=sides, hold_reasons=[])
    first = N(error=None, connection=N(binding=object()), native=N(pending=[]), closed=False)
    first.arrival_ledger = N(arrivals=(), applied=(), acknowledgements=())
    second = N(error=None, connection=N(binding=object()), native=N(pending=[]), closed=False)
    return N(evaluation_flags=N(closed=False, error=None, journal=journal, latest=flags),
             witness=N(journal=journal, pair=lambda frame: steps),
             state=dict(provisional_context_observer=N(rows=[row]), probabilistic_tracking_mode=first), mode=second)


def test_ready_preserves_next_legacy_exception() -> None:
    assert E.reasons(session(), FRAME) == ()


@pytest.mark.parametrize('change', ['chain', 'effect', 'origin', 'grace', 'basis', 'native', 'arrival', 'generation'])
def test_normal_holds_are_explicit(change: str) -> None:
    value = session()
    flag = value.evaluation_flags.latest['2P']
    if change == 'chain': flag['state'] = 'chain'
    elif change == 'effect': flag['effect_window'] = True
    elif change == 'origin': flag['origin_present'] = True
    elif change == 'grace': flag['grace_end'] = FRAME/60 + 1
    elif change == 'basis': value.mode = None
    elif change == 'native': value.mode.native.pending = ['人工未消費']
    elif change == 'arrival': value.mode.arrival_ledger = N(arrivals=(1,), applied=(), acknowledgements=(1,))
    else: value.witness.pair(FRAME)[1]['generation_after']['reset_epoch'] += 1
    assert E.reasons(value, FRAME)


@pytest.mark.parametrize('change', ['flag_error', 'source', 'call', 'context', 'unexpected_hold', 'probability', 'mode_error', 'ledger_missing'])
def test_internal_failure_is_not_normal_hold(change: str) -> None:
    value = session()
    row = value.state['provisional_context_observer'].rows[-1]
    if change == 'flag_error': value.evaluation_flags.error = RuntimeError('元障害')
    elif change == 'source': value.evaluation_flags.latest['1P']['scope']['run_id'] = 'foreign'
    elif change == 'call': value.evaluation_flags.latest['1P']['token'] = 'foreign'
    elif change == 'context': row['failures'] = ['元障害']
    elif change == 'unexpected_hold': row['sides']['1P']['hold_reasons'] = ['pb_missing']
    elif change == 'probability': row['sides']['1P']['before_hold']['probability']['errors'] = ['invalid']
    elif change == 'mode_error': value.mode.error, value.mode.closed = RuntimeError('元障害'), True
    else: value.state['probabilistic_tracking_mode'].arrival_ledger = None
    with pytest.raises(ValueError): E.reasons(value, FRAME)
