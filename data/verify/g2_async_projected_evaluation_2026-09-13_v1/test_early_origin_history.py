"""原J/Witnessの早期保存→元emit復元→後発W→同consumerを人工CPUで接続する。"""
from contextlib import ExitStack
import io
import json
from typing import Any
import pytest
from src import chain_prediction_ledger_v1 as L
from src.board import Board
from src.chain import ChainSimulator
import journal_witness as W
import journal_pair_reader as R
from journal_origin_capture import OriginCapture
from early_origin_history import EarlyHistory, CLOSE_NAME
from test_journal_pair_reader import setup
from test_journal_origin_capture import emit_pair, raw_event


def test_early_witness_handoff_and_original_live_continuation(tmp_path: Any) -> None:
    journal, pipe, _ = setup()
    journal.expected = []
    calls: list[int] = []
    with ExitStack() as stack:
        history = EarlyHistory(stack, journal, (100, 102, 104), tmp_path, W, R)
        for frame in (100, 102):
            emit_pair(journal, pipe, frame, raw_event(journal, 'landing' if frame == 100 else 'formula_read'))
            history.observe(pipe, frame)
        with pytest.raises(ValueError, match='J_original_emit'):
            W.install(ExitStack(), journal)
        history.seal(102)
        later = W.install(stack, journal)
        capture = OriginCapture(later, journal, pipe, L, ChainSimulator(), Board.from_dict,
            lambda raw: False, io.StringIO(), projected_input=lambda frame: calls.append(frame))
        stack.callback(capture.close)
        receipt = history.prime(capture, 102)
        assert receipt['recorded_first'] == 100 and receipt['available_frame'] == 102 and not calls
        root = capture.ledger.snapshot(capture.handles['2P']).predictions[0]
        assert root.available_at.frame_idx == 102
        assert 'recorded_100_received_102' in capture.ledger.snapshot(capture.handles['2P']).episodes[0].capture_source
        with pytest.raises(ValueError, match='transfer_lifetime'): history.prime(capture, 102)
        emit_pair(journal, pipe, 104, raw_event(journal, 'formula_read'))
        value = capture.completed(104)
        assert value['sides'][1]['origin']['available_at']['frame_idx'] == 102 and calls == [104]
        assert history.transferred and not history.records
    saved = json.loads((tmp_path / CLOSE_NAME).read_text())
    assert history.closed and later.closed and saved['cleanup_errors'] == []
    assert history.journal is None and history.pipe is None


def test_changed_buffer_is_rejected_before_ledger_mutation(tmp_path: Any) -> None:
    journal, pipe, _ = setup()
    journal.expected = []
    with ExitStack() as stack:
        history = EarlyHistory(stack, journal, (100,), tmp_path, W, R)
        emit_pair(journal, pipe, 100, raw_event(journal))
        history.observe(pipe, 100)
        history.seal(100)
        later = W.install(stack, journal)
        capture = OriginCapture(later, journal, pipe, L, ChainSimulator(), Board.from_dict,
            lambda raw: False, io.StringIO())
        history.records[0] += ' '
        with pytest.raises(ValueError, match='buffer_changed'): history.prime(capture, 100)
        assert not capture.handles and not history.transferred
        capture.close()


def test_original_body_error_preserved_and_partial_history_saved(tmp_path: Any) -> None:
    journal, pipe, _ = setup()
    journal.expected = []
    error = RuntimeError('original_driver_failure')
    with pytest.raises(RuntimeError) as caught:
        with ExitStack() as stack:
            history = EarlyHistory(stack, journal, (100,), tmp_path, W, R)
            emit_pair(journal, pipe, 100, raw_event(journal))
            history.observe(pipe, 100)
            raise error
    assert caught.value is error and history.closed
    saved = json.loads((tmp_path / CLOSE_NAME).read_text())
    assert saved['updates'] == 1 and saved['original_error'] == repr(error)
    assert not saved['transferred'] and saved['cleanup_errors'] == []


@pytest.mark.parametrize('fault', ['owner', 'clock', 'saved'])
def test_transfer_rejects_invalid_source_before_consumption(tmp_path: Any, fault: str) -> None:
    journal, pipe, _ = setup()
    journal.expected = []
    with ExitStack() as stack:
        history = EarlyHistory(stack, journal, (100,), tmp_path, W, R)
        emit_pair(journal, pipe, 100, raw_event(journal))
        history.observe(pipe, 100)
        history.seal(100)
        later = W.install(stack, journal)
        capture = OriginCapture(later, journal, pipe, L, ChainSimulator(), Board.from_dict,
            lambda raw: False, io.StringIO())
        if fault == 'owner': capture.pipe = object()
        elif fault == 'clock': journal.history.frame += 2
        else:
            with (tmp_path / 'EARLY_ORIGIN_HISTORY.jsonl').open('a') as stream: stream.write('corruption')
        with pytest.raises(ValueError, match='early_history_'): history.prime(capture, 100)
        assert not capture.handles and not history.transfer_started
        capture.close()


def test_prime_failure_is_saved_once_and_keeps_original_error(tmp_path: Any) -> None:
    journal, pipe, _ = setup()
    journal.expected = []
    error = RuntimeError('history_simulation_failure')
    class Broken:
        def simulate(self, board: Any) -> Any: raise error
    sink = io.StringIO()
    with pytest.raises(RuntimeError) as caught:
        with ExitStack() as stack:
            history = EarlyHistory(stack, journal, (100,), tmp_path, W, R)
            emit_pair(journal, pipe, 100, raw_event(journal))
            history.observe(pipe, 100)
            history.seal(100)
            later = W.install(stack, journal)
            capture = OriginCapture(later, journal, pipe, L, Broken(), Board.from_dict, lambda raw: False, sink)
            stack.callback(capture.close)
            history.prime(capture, 100)
    assert caught.value is error and history.transfer_started and not history.transferred
    assert json.loads(sink.getvalue())['kind'] == 'origin_capture_failure'
    saved = json.loads((tmp_path / CLOSE_NAME).read_text())
    assert saved['error'] == repr(error) and saved['original_error'] == repr(error)
