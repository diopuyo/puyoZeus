"""起点→遅れてPB→seal/prime→現在参考が成立する反証。PB供給器だけ人工。"""
from contextlib import ExitStack
from dataclasses import replace
import io
import json
from typing import Any
import stream_witness as W
import writer_contract_v2 as K
import journal_pair_reader as R
import fixed_origin_reference as F
from early_origin_history import EarlyHistory
from journal_origin_capture import OriginCapture
from test_fixed_origin_reference import emit
from test_journal_pair_reader import setup
from test_diagnose_video38_prediction_ledger_shadow_v1 import _event, _board
from src import chain_prediction_ledger_v1 as L
from src.board import Board
from src.chain import ChainSimulator
from settled_notice import is_settled


class ArtificialProbability:
    def __init__(self, stack: Any) -> None:
        self.closed = False
        stack.callback(setattr, self, 'closed', True)

    def snapshot(self, row: dict) -> dict:
        assert not self.closed
        return dict(frame=row['frame_idx'], candidate=row['frame_idx'] == 102)

    @staticmethod
    def is_candidate(value: dict) -> bool:
        return value['candidate']


def test_later_probability_is_available_before_session_without_window_extension(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(W, 'K', K)
    monkeypatch.setattr(R, 'W', W)
    journal, pipe, _ = setup()
    journal.expected = []
    landing = replace(_event(_board(), 'landing'), trigger_sec=100 / F.FPS, end_sec=100 / F.FPS)
    with ExitStack() as stack:
        history = EarlyHistory(stack, journal, (100, 102, 104), tmp_path, W, R,
                               probability_factory=ArtificialProbability)
        for frame in history.frames:
            event = landing if frame == 100 else replace(landing, mechanism='formula_read')
            emit(journal, pipe, frame, 'CHAIN', event)
            history.observe(pipe, frame)
        history.seal(104)
        later = W.install(stack, journal)
        capture = OriginCapture(later, journal, pipe, L, ChainSimulator(), Board.from_dict, is_settled, io.StringIO())
        stack.callback(capture.close)
        history.prime(capture, 104)
        emit(journal, pipe, 106, 'CHAIN', replace(landing, mechanism='formula_read'))
        capture.completed(106)
        reference = F.read(capture, '2P', 106)
        assert reference['status'] == 'REFERENCE_ONLY'
        assert json.loads(history.probability_inputs()[0])['frame'] == 102
        assert reference['origin']['available_at']['frame_idx'] == history.last_frame == 104
        assert landing.trigger_sec == 100 / F.FPS and history.probability.closed
