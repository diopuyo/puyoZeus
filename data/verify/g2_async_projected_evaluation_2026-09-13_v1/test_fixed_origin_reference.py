"""原J/Stream/consumerを通した参考rootの読取・一時保留・別landing拒否。"""
from contextlib import ExitStack
from dataclasses import replace
import io
import sys
from types import SimpleNamespace as N
from typing import Any
import pytest
import stream_witness as W
import writer_contract_v2 as K
import journal_pair_reader as R
import fixed_origin_reference as F
from journal_origin_capture import OriginCapture
from test_journal_pair_reader import setup
from test_diagnose_video38_prediction_ledger_shadow_v1 import _event, _board
from src import chain_prediction_ledger_v1 as L
from src.board import Board
from src.chain import ChainSimulator
from settled_notice import is_settled


def emit(journal: Any, pipe: Any, frame: int, state: str, event: Any, *, observe: bool = True) -> None:
    journal.history.frame, journal.history.time_sec = frame, frame / F.FPS
    journal.expected.extend([(frame, side) for side in R.SIDES])
    journal.selected = set(journal.expected)
    pipe._active_chain_2p = event
    raw = type(journal).complete_step.__globals__['event'](event)
    for side in R.SIDES:
        scope = journal.scope(pipe, side, frame, frame / F.FPS)
        events = [dict(stage='origin_after', active_origin=raw)] if side == '2P' and observe else []
        item = dict(scope=scope, token=f'step:{journal.steps}', events=events, epoch=journal.epoch(pipe, side),
                    return_line=None, frame=None, pipe=pipe)
        journal.steps += 1
        result = None if side == '1P' else N(confirmed_board=None, inferred_board=None, state=N(name=state))
        journal.complete_step(item, result, None, sys.getprofile())


@pytest.fixture
def source(monkeypatch: Any) -> Any:
    monkeypatch.setattr(W, 'K', K)
    monkeypatch.setattr(R, 'W', W)
    journal, pipe, values = setup()
    journal.expected = []
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        capture = OriginCapture(witness, journal, pipe, L, ChainSimulator(), Board.from_dict, is_settled, io.StringIO())
        stack.callback(capture.close)
        landing = replace(_event(_board(), 'landing'), trigger_sec=100 / F.FPS, end_sec=100 / F.FPS)
        emit(journal, pipe, 100, 'CHAIN', landing)
        capture.completed(100)
        yield capture, journal, pipe, values, landing


def test_reference_hold_resume_settled_and_distinct_landing(source: Any) -> None:
    capture, journal, pipe, _, landing = source
    before = capture.ledger.snapshot(capture.handles['2P'])
    original_bytes = capture.witness.original.getvalue()
    first = F.read(capture, '2P', 100)
    assert first['status'] == 'REFERENCE_ONLY' and not first['future_fire_power_supply_authorized']
    first['origin']['chain_count'] = -1
    assert capture.ledger.snapshot(capture.handles['2P']) == before
    assert capture.witness.original.getvalue() == original_bytes
    emit(journal, pipe, 102, 'GRAVITY_SETTLE', None)
    capture.completed(102)
    assert F.read(capture, '2P', 102)['reason'] == 'returned_origin_missing'
    formula = replace(landing, mechanism='formula_read', trigger_sec=104 / F.FPS)
    emit(journal, pipe, 104, 'CHAIN', formula)
    capture.completed(104)
    assert F.read(capture, '2P', 104)['status'] == 'REFERENCE_ONLY'
    emit(journal, pipe, 106, 'STABLE', replace(formula, mechanism='baseline'))
    capture.completed(106)
    assert F.read(capture, '2P', 106)['reason'] == 'returned_origin_is_settled_notice'
    emit(journal, pipe, 108, 'CHAIN', replace(landing, trigger_sec=108 / F.FPS))
    capture.completed(108)
    assert F.read(capture, '2P', 108)['reason'] == 'multiple_landing_contents_in_same_instance'
    assert capture.handles['2P'].instance_id == before.handle.instance_id


@pytest.mark.parametrize('fault', ['clock', 'scope', 'closed', 'error'])
def test_invalid_read_is_rejected(source: Any, fault: str) -> None:
    capture, journal, _, _, _ = source
    if fault == 'clock': journal.history.frame += 2
    elif fault == 'scope': capture.scopes['2P']['source_id'] = 'foreign'
    elif fault == 'closed': capture.closed = True
    else: capture.error = RuntimeError('original')
    with pytest.raises(ValueError): F.read(capture, '2P', 100)


def test_generation_retirement_and_unobserved_origin_hold(source: Any) -> None:
    capture, journal, pipe, values, landing = source
    emit(journal, pipe, 102, 'CHAIN', replace(landing, trigger_sec=102 / F.FPS), observe=False)
    capture.completed(102)
    assert F.read(capture, '2P', 102)['reason'] == 'returned_origin_not_consumed'
    values['2P'].action_revision += 1
    emit(journal, pipe, 104, 'CHAIN', landing)
    capture.completed(104)
    assert F.read(capture, '2P', 104)['reason'] == 'fixed_origin_handle_missing'
