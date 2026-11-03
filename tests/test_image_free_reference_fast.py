"""履歴の追記・過去値訂正・CP復元でも時刻の値と並びを変えない。"""
from dataclasses import asdict
import pickle
import struct
from types import SimpleNamespace

from scripts.image_free_reference_fast import reference_state
from src.exchange_event_tracker import ExchangeChainRecord, ExchangeRecord
from src.phase_j.live_b3 import reference_state as original


def bits(value: dict) -> tuple:
    return value['source'], value['probability'], tuple(struct.pack('d', v) for v in value['physical'])


def test_mutable_history_and_checkpoint_keep_exact_reference() -> None:
    chain = ExchangeChainRecord('1P', 1, 2., 3.)
    record = ExchangeRecord(1, 1, -0., chains=[chain])
    tracker = SimpleNamespace(records=[record], current=record, source='S1', probability=.3)
    for frame in range(30):
        record.values.append(dict(t_sec=frame, ignored=[list(range(100))]*10))
        chain.end_signal_sec = None if frame % 3 else frame+.25
        record.landings.append(dict(fall_start_sec=frame, t_sec=frame+.5, unused='history'))
        record.closed_sec = None if frame < 20 else 20.
        assert bits(reference_state(tracker)) == bits(original(tracker))
        if frame == 15:
            tracker = pickle.loads(pickle.dumps(tracker))
            record, chain = tracker.records[0], tracker.records[0].chains[0]
    tracker.records.append(asdict(record))
    tracker.current = None
    tracker.records.reverse()
    assert bits(reference_state(tracker)) == bits(original(tracker))


def test_empty_and_past_record_mutation_are_not_cached() -> None:
    tracker = SimpleNamespace(records=[], current=None, source=None, probability=None)
    assert reference_state(tracker) == original(tracker)
    old = ExchangeRecord(1, 0, 1., closed_sec=2.)
    tracker.records = [old, ExchangeRecord(2, 1, 5.)]
    assert bits(reference_state(tracker)) == bits(original(tracker))
    old.closed_sec = 9.
    assert bits(reference_state(tracker)) == bits(original(tracker))
