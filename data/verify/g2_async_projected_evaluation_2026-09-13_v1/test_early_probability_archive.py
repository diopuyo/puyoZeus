"""履歴の保存・移管・対象外後の候補保持を人工PB供給器で限定検証する。"""
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


class ArtificialProbability:
    def __init__(self, stack: Any) -> None:
        self.closed = False
        stack.callback(setattr, self, 'closed', True)

    def snapshot(self, step: dict) -> dict:
        assert not self.closed and step['side'] == '2P'
        return dict(frame=step['frame_idx'], candidate=step['frame_idx'] == 100,
                    basis_registered=False, quality_gate_clear=False)

    @staticmethod
    def is_candidate(value: dict) -> bool:
        return value['candidate']


@pytest.mark.parametrize('tamper', [False, True])
def test_candidate_survives_later_hold_and_one_time_prime(tmp_path: Any, tamper: bool) -> None:
    journal, pipe, _ = setup()
    journal.expected = []
    with ExitStack() as stack:
        history = EarlyHistory(stack, journal, (100, 102), tmp_path, W, R,
                               probability_factory=ArtificialProbability)
        probability = history.probability
        for frame in (100, 102):
            emit_pair(journal, pipe, frame, raw_event(journal))
            history.observe(pipe, frame)
        with pytest.raises(ValueError, match='inputs_lifetime'): history.probability_inputs()
        history.seal(102)
        assert probability.closed and len(history.probability_records) == 1
        later = W.install(stack, journal)
        capture = OriginCapture(later, journal, pipe, L, ChainSimulator(), Board.from_dict,
                                lambda raw: False, io.StringIO())
        stack.callback(capture.close)
        history.prime(capture, 102)
        assert not history.records and history.transferred
        if tamper:
            history.probability_records = (history.probability_records[0] + ' ',)
            with pytest.raises(ValueError, match='inputs_changed'): history.probability_inputs()
        else:
            assert json.loads(history.probability_inputs()[0])['frame'] == 100
            saved = [json.loads(line) for line in (tmp_path / 'EARLY_ORIGIN_HISTORY.jsonl').read_text().splitlines()]
            assert saved[0]['probability']['candidate'] and not saved[1]['probability']['candidate']
    assert history.probability is None and history.probability_records == ()
    assert json.loads((tmp_path / CLOSE_NAME).read_text())['probability_candidates'] == 1
    with pytest.raises(ValueError, match='inputs_lifetime'): history.probability_inputs()
