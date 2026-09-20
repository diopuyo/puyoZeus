"""実Sessionが選ぶ既存stream WitnessをRへ接続する反例・正常対照。"""
from contextlib import ExitStack
from typing import Any
import stream_witness as W
import writer_contract_v2 as K
import journal_pair_reader as R
from test_journal_pair_reader import setup
from test_journal_origin_capture import emit_pair, raw_event


def test_original_stream_witness_is_accepted_by_pair_reader(monkeypatch: Any) -> None:
    monkeypatch.setattr(W, 'K', K)
    monkeypatch.setattr(R, 'W', W)
    journal, pipe, _ = setup()
    journal.expected = []
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        emit_pair(journal, pipe, 100, raw_event(journal))
        result = R.read_pair(witness, journal, pipe, 100)
        assert not result.holds and result.frame == 100
        assert witness.error is None
