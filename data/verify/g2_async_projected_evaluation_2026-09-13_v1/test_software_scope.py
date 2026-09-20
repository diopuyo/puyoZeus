"""tracker世代不変の原J epoch変更を、元Witness経由で検査する。"""
from contextlib import ExitStack
import io
import json
import pytest
from src import chain_prediction_ledger_v1 as L
from src.board import Board
from src.chain import ChainSimulator
from test_journal_pair_reader import setup
from test_journal_origin_capture import emit_pair, raw_event
import journal_witness as W
from journal_origin_capture import OriginCapture


def test_independent_epoch_retires_old_origin_then_accepts_fresh_landing() -> None:
    journal, pipe, values = setup()
    journal.expected = []
    sink = io.StringIO()
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        capture = OriginCapture(witness, journal, pipe, L, ChainSimulator(), Board.from_dict,
                                lambda raw: False, sink)
        raw = raw_event(journal)
        emit_pair(journal, pipe, 100, raw)
        first = capture.completed(100)
        old_handle = capture.handles['2P']
        old_generation = capture.ledger.snapshot(old_handle).generation
        journal.controller.instances[id(pipe)].histories['2P'].epoch = 1
        emit_pair(journal, pipe, 102, raw)
        original = journal.stream.getvalue()
        second = capture.completed(102)
        assert second['sides'][1]['origin'] is None
        assert second['sides'][1]['reasons'] == ['scope_changed_generation_not_advanced']
        assert journal.stream.getvalue() == original
        assert capture.ledger.snapshot(old_handle).generation == old_generation
        assert second['sides'][0] == first['sides'][0]
        fresh = raw_event(journal)
        fresh.update(trigger_sec=104 / 60, end_sec=104 / 60)
        emit_pair(journal, pipe, 104, fresh)
        third = capture.completed(104)
        assert third['sides'][1]['origin'] is None and capture.error is None
        assert third['sides'][1]['source_scope']['software_reset'] == 1
        assert values['2P'].action_revision == 0
        values['2P'].action_revision = 1
        emit_pair(journal, pipe, 106, raw)
        retired = capture.completed(106)
        assert retired['sides'][1]['reasons'] == ['origin_from_retired_software_scope']
        fresh = raw_event(journal)
        fresh.update(trigger_sec=108 / 60, end_sec=108 / 60)
        emit_pair(journal, pipe, 108, fresh)
        recovered = capture.completed(108)
        assert recovered['sides'][1]['origin'] is not None
        assert recovered['sides'][1]['instance_id'] != first['sides'][1]['instance_id']
        packets = [json.loads(line) for line in sink.getvalue().splitlines()]
        assert len(packets) == 5 and all(not p['future_fire_power_supply_authorized'] for p in packets)
        capture.close()
        assert not capture.scopes and not capture.origin_generations and capture.ledger is None


@pytest.mark.parametrize('fault', ['source_id', 'software_reset'])
def test_scope_owner_change_and_rewind_fail_without_origin_replacement(fault: str) -> None:
    journal, pipe, _ = setup()
    journal.expected = []
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        capture = OriginCapture(witness, journal, pipe, L, ChainSimulator(), Board.from_dict,
                                lambda raw: False, io.StringIO())
        journal.controller.instances[id(pipe)].histories['2P'].epoch = 1
        emit_pair(journal, pipe, 100, raw_event(journal))
        first = capture.completed(100)
        first['sides'][1]['source_scope']['software_reset'] = 999
        assert capture.scopes['2P']['software_reset'] == 1
        handle = capture.handles['2P']
        before = capture.ledger.snapshot(handle)
        if fault == 'source_id': journal.source_id = 'replaced-source'
        else: journal.controller.instances[id(pipe)].histories['2P'].epoch = 0
        emit_pair(journal, pipe, 102, raw_event(journal))
        with pytest.raises(ValueError, match='source_owner_changed|scope_rewind'):
            capture.completed(102)
        assert capture.error is not None and capture.ledger.snapshot(handle) == before
        capture.close()


def test_landing_first_seen_during_hold_cannot_cross_generation() -> None:
    journal, pipe, values = setup()
    journal.expected = []
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        capture = OriginCapture(witness, journal, pipe, L, ChainSimulator(), Board.from_dict,
                                lambda raw: False, io.StringIO())
        emit_pair(journal, pipe, 100, raw_event(journal))
        capture.completed(100)
        journal.controller.instances[id(pipe)].histories['2P'].epoch = 1
        held = raw_event(journal)
        held.update(trigger_sec=101 / 60, end_sec=101 / 60)
        emit_pair(journal, pipe, 102, held)
        assert capture.completed(102)['sides'][1]['origin'] is None
        values['2P'].action_revision = 1
        emit_pair(journal, pipe, 104, held)
        final = capture.completed(104)['sides'][1]
        assert final['origin'] is None
        assert final['reasons'] == ['origin_from_retired_software_generation']
        capture.close()


def test_scope_change_before_any_landing_allows_truly_new_origin() -> None:
    journal, pipe, values = setup()
    journal.expected = []
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        capture = OriginCapture(witness, journal, pipe, L, ChainSimulator(), Board.from_dict,
                                lambda raw: False, io.StringIO())
        emit_pair(journal, pipe, 100, raw_event(journal, 'formula_read'))
        assert capture.completed(100)['sides'][1]['origin'] is None
        journal.controller.instances[id(pipe)].histories['2P'].epoch = 1
        fresh = raw_event(journal)
        fresh.update(trigger_sec=102 / 60, end_sec=102 / 60)
        emit_pair(journal, pipe, 102, fresh)
        final = capture.completed(102)['sides'][1]
        assert final['origin'] is not None and final['source_scope']['software_reset'] == 1
        assert values['2P'].action_revision == 0
        assert not capture.scope_holds
        capture.close()
