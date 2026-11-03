"""時刻投影の等価性と不要な深いコピーの排除。"""
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.exchange_event_tracker import ExchangeChainRecord, ExchangeRecord
from src.phase_j.live_b3 import reference_state


@pytest.mark.parametrize('mapping', [False, True])
def test_reference_times_equal(mapping: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    record = ExchangeRecord(1, 2, 3., chains=[ExchangeChainRecord('1P', 1, 3., 4.)],
                            landings=[dict(fall_start_sec=5., t_sec=6.)],
                            values=[dict(p1=.2)]*100, closed_sec=7.)
    value = asdict(record) if mapping else record
    tracker = SimpleNamespace(records=[value], current=value, source='death', probability=0.)
    monkeypatch.setenv('PUYO_LIVE_REFERENCE_TIMES_ONLY', '0')
    before = reference_state(tracker)
    monkeypatch.setenv('PUYO_LIVE_REFERENCE_TIMES_ONLY', '1')
    with patch('src.phase_j.live_b3.asdict', side_effect=AssertionError('不要な複製')):
        assert reference_state(tracker) == before
    assert before['physical'] == [3., 4., 5., 6., 7.]
    if not mapping:
        record.chains[0].end_signal_sec = 8.
        assert reference_state(tracker)['physical'][-1] == 8.
    tracker.records, tracker.current = [], None
    assert reference_state(tracker)['physical'] == []
