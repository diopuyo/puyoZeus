"""元J完了→既存Witness→既存予測台帳→保存/解放を人工CPUで接続する。"""
from contextlib import ExitStack
from dataclasses import replace
import io
import json
from typing import Any
import pytest
from src import chain_prediction_ledger_v1 as L
from src.board import Board
from src.chain import ChainSimulator
from test_diagnose_video38_prediction_ledger_shadow_v1 import _event, _board
from test_journal_pair_reader import setup, complete
import journal_witness as W
from journal_origin_capture import OriginCapture


def emit_pair(journal: Any, pipe: Any, frame: int, raw: dict) -> None:
    journal.history.frame, journal.history.time_sec = frame, frame / 60
    ordinal = journal.steps
    journal.expected.extend([(frame, side) for side in ('1P', '2P')])
    journal.selected = set(journal.expected)
    complete(journal, pipe, '1P', ordinal, frame=frame)
    complete(journal, pipe, '2P', ordinal + 1, frame=frame,
             events=[dict(stage='origin_after', active_origin=raw)])


def raw_event(journal: Any, mechanism: str = 'landing') -> dict:
    event = replace(_event(_board(), mechanism), trigger_sec=100 / 60, end_sec=100 / 60)
    return type(journal).complete_step.__globals__['event'](event)


def test_original_J_to_ledger_save_duplicate_and_generation_replacement() -> None:
    journal, pipe, values = setup()
    journal.expected = []
    sink = io.StringIO()
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        capture = OriginCapture(witness, journal, pipe, L, ChainSimulator(), Board.from_dict,
                                lambda raw: False, sink)
        raw = raw_event(journal)
        emit_pair(journal, pipe, 100, raw)
        before = journal.stream.getvalue()
        first = capture.completed(100)
        assert first['sides'][1]['origin']['is_origin_reference']
        assert journal.stream.getvalue() == before
        with pytest.raises(ValueError, match='duplicate'):
            capture.completed(100)
        emit_pair(journal, pipe, 102, raw_event(journal, 'formula_read'))
        second = capture.completed(102)
        assert second['sides'][1]['origin'] == first['sides'][1]['origin']
        values['2P'].action_revision = 1
        fresh = raw_event(journal)
        fresh.update(trigger_sec=104 / 60, end_sec=104 / 60)
        emit_pair(journal, pipe, 104, fresh)
        third = capture.completed(104)
        assert third['sides'][1]['instance_id'] != first['sides'][1]['instance_id']
        packets = [json.loads(line) for line in sink.getvalue().splitlines()]
        assert len(packets) == 3 and all(not p['future_fire_power_supply_authorized'] for p in packets)
        capture.close()
        assert not capture.handles and not capture.seen
        assert capture.ledger is None and capture.pipe is None and capture.witness is None
        with pytest.raises(ValueError, match='lifetime'):
            capture.completed(106)


def test_source_hash_failure_is_sticky_without_original_stream_changes() -> None:
    journal, pipe, _ = setup()
    journal.expected = []
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        sink = io.StringIO()
        capture = OriginCapture(witness, journal, pipe, L, ChainSimulator(), Board.from_dict,
                                lambda raw: False, sink)
        raw = raw_event(journal)
        raw['before_board']['sha256'] = 'wrong'
        emit_pair(journal, pipe, 100, raw)
        before = journal.stream.getvalue()
        with pytest.raises(ValueError, match='source_board_hash'):
            capture.completed(100)
        assert capture.error is not None and journal.stream.getvalue() == before
        failure = json.loads(sink.getvalue())
        assert failure['kind'] == 'origin_capture_failure' and failure['source_rows_json'] is not None
        with pytest.raises(ValueError, match='lifetime'):
            capture.completed(102)


def test_writer_failure_preserves_original_error_and_records_save_failure() -> None:
    journal, pipe, _ = setup()
    journal.expected = []
    error = OSError('artificial_disk_failure')
    class BrokenWriter:
        def write(self, value: str) -> None:
            raise error
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        capture = OriginCapture(witness, journal, pipe, L, ChainSimulator(), Board.from_dict,
                                lambda raw: False, BrokenWriter())
        emit_pair(journal, pipe, 100, raw_event(journal))
        with pytest.raises(OSError) as caught:
            capture.completed(100)
        assert caught.value is error and capture.error is error and capture.save_error is error
        capture.close()


def test_actual_settled_classifier_does_not_reopen_or_erase_origin() -> None:
    from settled_notice import is_settled
    journal, pipe, _ = setup()
    journal.expected = []
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        capture = OriginCapture(witness, journal, pipe, L, ChainSimulator(), Board.from_dict,
                                is_settled, io.StringIO())
        landing = raw_event(journal)
        assert not is_settled(landing)
        emit_pair(journal, pipe, 100, landing)
        initial = capture.completed(100)['sides'][1]['origin']
        notice = raw_event(journal, 'baseline')
        assert is_settled(notice)
        emit_pair(journal, pipe, 102, notice)
        assert capture.completed(102)['sides'][1]['origin'] == initial
        assert len(capture.ledger.snapshot(capture.handles['2P']).episodes) == 1
        capture.close()
