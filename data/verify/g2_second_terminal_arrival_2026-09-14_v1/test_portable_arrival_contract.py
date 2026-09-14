"""PR向けの最小契約対照。動画/モデル/凍結snapshotなし、原Native認証は対象外。"""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
import second_arrival_state as A
import warning_source as W

ROOT = Path(__file__).resolve().parent
SCOPE = ('synthetic-source', 'portable-contract', 0, 1, 2, 0, '2P')
START, DEADLINE = 100, 110
PAIR_A, PAIR_B = (5, 3), (3, 4)


@pytest.fixture
def ledger() -> Any:
    path = ROOT.parent / 'g2_arrival_ack_candidate_2026-09-12_v1/ledger.py'
    alias = '_portable_arrival_contract_ledger'
    assert alias not in sys.modules
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    with ExitStack() as stack:
        sys.modules[alias] = module
        stack.callback(sys.modules.pop, alias)
        spec.loader.exec_module(module)
        yield module


def row(frame: int, tokens: list, pairs: list, added: list) -> dict:
    return dict(kind='enqueue', source_id=SCOPE[0], run_id=SCOPE[1], software_reset=SCOPE[2],
        pipe_object_id=SCOPE[3], generation=dict(reset_epoch=SCOPE[5]), side=SCOPE[-1],
        frame_idx=frame, time_sec=frame/A.FPS, status='returned', active=True, token=f'enqueue:{frame}',
        fifo_occurrence_tokens=tokens, added_occurrence_tokens=added,
        after=dict(pending_tsumo=[list(pair) for pair in pairs]))


def event(token: str, pair: tuple, frame: int) -> Any:
    return N(occurrence_token=token, pair=pair, consumed_frame=frame, source_call_token=f'ack:{frame}')


def test_applied_before_late_ack_is_not_reapplied(ledger: Any) -> None:
    initial = ledger.Ledger(SCOPE, START, DEADLINE, START)
    rows = (row(102, ['A'], [PAIR_A], ['A']), row(104, ['A', 'B'], [PAIR_A, PAIR_B], ['B']))
    original = deepcopy(rows)
    arrived = A.advance(ledger, initial, rows, (event('A', PAIR_A, 104),), 104)
    applied = ledger.applied(arrived, ('A', 'B'), 104)
    assert len(applied.arrivals) == len(applied.applied) == 2 and len(applied.acknowledgements) == 1
    final = A.advance(ledger, applied, (row(106, ['B'], [PAIR_B], []),), (event('B', PAIR_B, 106),), 106)
    assert ledger.drained(final) and final.applied == applied.applied and rows == original
    assert initial.clock == START and not initial.arrivals
    with pytest.raises(ValueError):
        ledger.applied(final, ('B',), 108)


@pytest.mark.parametrize('bad', ['scope', 'gap', 'fifo'])
def test_invalid_source_is_rejected(ledger: Any, bad: str) -> None:
    initial = ledger.Ledger(SCOPE, START, DEADLINE, START)
    value = row(102, ['A'], [PAIR_A], ['A'])
    if bad == 'scope':
        value['side'] = '1P'
    elif bad == 'gap':
        value['frame_idx'] = 104
    else:
        value['fifo_occurrence_tokens'] = []
    with pytest.raises(ValueError):
        A.advance(ledger, initial, (value,), (), 102)
    assert not initial.arrivals


def warning(frame: int) -> dict:
    call = f'synthetic:{frame}'
    return dict(scope=SCOPE, frame=frame, call_token=call, context_known=True, active=True,
        own_chain_active=False, own_score_delta=0, active_origin=False, chain_event=False,
        confirmed_has_ojama=frame == 104, observation=dict(status='POSITIVE', scope=SCOPE,
            frame=frame, call_token=call, image_sha256='synthetic-not-a-camera-hash', conditional_lower_bound=30))


def test_warning_condition_is_not_future_guarantee(ledger: Any) -> None:
    rows = (warning(102), warning(104))
    result = W.select(ledger.require, rows, SCOPE, START, 104)
    assert result['conditional_drop_amount'] == 30
    assert not result['calibrated'] and not result['future_landing_guaranteed']
    rows[-1]['own_score_delta'] = 40
    with pytest.raises(ValueError, match='warning_cancellation'):
        W.select(ledger.require, rows, SCOPE, START, 104)
