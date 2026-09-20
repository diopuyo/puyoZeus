"""退役世代のlandingが新instanceへ再登録されないことを検査する。"""
from contextlib import ExitStack
import io
from src import chain_prediction_ledger_v1 as L
from src.board import Board
from src.chain import ChainSimulator
from test_journal_pair_reader import setup
from test_journal_origin_capture import emit_pair, raw_event
from journal_origin_capture import OriginCapture
import journal_witness as W


def test_old_landing_cannot_be_reopened_after_software_generation_change() -> None:
    journal, pipe, values = setup()
    journal.expected = []
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        capture = OriginCapture(witness, journal, pipe, L, ChainSimulator(), Board.from_dict,
                                lambda raw: False, io.StringIO())
        old = raw_event(journal)
        emit_pair(journal, pipe, 100, old)
        first = capture.completed(100)['sides'][1]
        values['2P'].action_revision += 1
        emit_pair(journal, pipe, 102, old)
        second = capture.completed(102)['sides'][1]
        assert second['instance_id'] is None and second['origin'] is None
        assert second['reasons'] == ['origin_from_retired_software_generation']
        capture.close()
