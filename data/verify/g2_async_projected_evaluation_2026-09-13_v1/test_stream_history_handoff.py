"""原Provider/adoption二中継を保持したstream早期履歴のCPU結合検査。"""
from contextlib import ExitStack
import io
import json
from typing import Any
import pytest
import stream_witness as W
import writer_contract_v2 as K
import reproduce_v2 as TWO
import journal_pair_reader as R
from src import chain_prediction_ledger_v1 as L
from src.board import Board
from src.chain import ChainSimulator
from early_origin_history import EarlyHistory, CLOSE_NAME
from journal_origin_capture import OriginCapture
from test_journal_pair_reader import setup
from test_journal_origin_capture import emit_pair, raw_event


def test_two_original_wrappers_survive_early_to_late_handoff(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(W, 'K', K)
    monkeypatch.setattr(R, 'W', W)
    journal, pipe, _ = setup()
    journal.expected = []
    monkeypatch.setattr(TWO.R.T, 'recorder', lambda: journal)
    attached, provider, controller, adoption = TWO.attached()
    assert attached is journal
    original_emit, original_stream = journal.emit, journal.stream
    with ExitStack() as stack:
        provider.attach(stack, controller)
        wrapped_emit = journal.emit
        history = EarlyHistory(stack, journal, (100, 102, 104), tmp_path, W, R)
        for frame in (100, 102):
            emit_pair(journal, pipe, frame, raw_event(journal, 'landing' if frame == 100 else 'formula_read'))
            history.observe(pipe, frame)
        with pytest.raises(ValueError, match='J_original_stream'):
            W.install(ExitStack(), journal)
        history.seal(102)
        assert journal.stream is original_stream and journal.emit is wrapped_emit
        later = W.install(stack, journal)
        capture = OriginCapture(later, journal, pipe, L, ChainSimulator(), Board.from_dict,
                                lambda raw: False, io.StringIO())
        stack.callback(capture.close)
        history.prime(capture, 102)
        emit_pair(journal, pipe, 104, raw_event(journal, 'formula_read'))
        value = capture.completed(104)
        assert value['sides'][1]['origin']['available_at']['frame_idx'] == 102
        assert journal.emit is wrapped_emit and history.transferred and not history.records
        assert later.error is None and not journal.errors
    assert journal.emit == original_emit and journal.stream is original_stream and adoption.closed
    assert later.closed and history.closed and history.journal is None
    saved = json.loads((tmp_path / CLOSE_NAME).read_text())
    assert saved['cleanup_errors'] == [] and saved['transferred']
