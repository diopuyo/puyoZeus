"""論理の待機、遅着、鮮度、境界と未来入力の門を独立に確かめる。"""
from collections import deque
from queue import Queue

import pytest

from src.phase_j import live_ex_async as A
from src.phase_j import live_ex_delay as D


def completed(serial: int = 1, **changes: object) -> dict:
    row = dict(serial=serial, epoch=4, key=(4, 2, (3, None)), fight=True,
               captured_at=10., requested_at=10., completed_at=10.05,
               input_t=3., eligible_at=10.2, extra={})
    return dict(row, **changes)


def owner(rows: list[dict]) -> D.DelayedAsyncEx:
    subject = object.__new__(D.DelayedAsyncEx)
    subject.delay_sec = .2
    subject.completed = deque(rows)
    subject.outputs = Queue()
    subject.serial, subject.epoch, subject.error = 3, 4, None
    subject.latest = None
    subject.key = dict(serial=3, key=(4, 2, (3, None)), fight=True)
    return subject


@pytest.mark.parametrize(('now', 'accepted'), [(10.199999, False), (10.2, True),
                                              (10.5, True), (10.500001, False)])
def test_logical_delay_and_freshness(monkeypatch: pytest.MonkeyPatch, now: float, accepted: bool) -> None:
    monkeypatch.setattr(D, 'perf_counter', lambda: now)
    subject = owner([completed()])
    assert (subject.available(now, 3.)[0] is not None) == accepted


def test_late_ipc_waits_for_actual_arrival(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 10.35
    monkeypatch.setattr(A, 'perf_counter', lambda: now)
    monkeypatch.setattr(D, 'perf_counter', lambda: now)
    subject = owner([])
    subject.outputs.put(dict(kind='result', value=completed()))
    subject.poll()
    result = subject.completed[0]
    assert result['eligible_at'] == result['received_at'] == now
    assert result['extra']['av_ex_p1'] == result['extra']['av_ex_p2'] == now
    assert subject.available(now-.001, 3.)[0] is None
    assert subject.available(now, 3.)[0] is result


def test_early_ipc_records_logical_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(A, 'perf_counter', lambda: 10.1)
    monkeypatch.setattr(D, 'perf_counter', lambda: 10.1)
    subject = owner([])
    subject.outputs.put(dict(kind='result', value=completed()))
    subject.poll()
    result = subject.completed[0]
    assert result['received_at'] == 10.1 and result['eligible_at'] == 10.2
    assert result['extra']['av_ex_p1'] == result['extra']['av_ex_p2'] == 10.2


def test_young_completion_does_not_erase_ready_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D, 'perf_counter', lambda: 10.3)
    old = completed()
    new = completed(2, requested_at=10.2, captured_at=10.2, eligible_at=10.4)
    subject = owner([old, new])
    assert subject.available(10.3, 3.)[0] is old
    assert subject.available(10.4, 3.)[0] is new


@pytest.mark.parametrize('change', [dict(epoch=5), dict(key=(4, 3, (3, None))),
                                    dict(input_t=3.1), dict(fight=False)])
def test_invalid_completion_never_enters_model(monkeypatch: pytest.MonkeyPatch, change: dict) -> None:
    monkeypatch.setattr(D, 'perf_counter', lambda: 10.3)
    subject = owner([completed(**change)])
    assert subject.available(10.3, 3.)[0] is None


@pytest.mark.parametrize('change', [dict(serial=2), dict(fight=False), dict(key=(4, 2, (4, None)))])
def test_latest_observation_invalidates_previous_exchange(monkeypatch: pytest.MonkeyPatch, change: dict) -> None:
    monkeypatch.setattr(D, 'perf_counter', lambda: 10.3)
    subject = owner([completed()])
    subject.key.update(change)
    assert subject.available(10.3, 3.)[0] is None


def test_no_hold_after_expiry_even_while_next_result_is_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D, 'perf_counter', lambda: 10.51)
    subject = owner([completed(), completed(2, captured_at=10.4, eligible_at=10.6)])
    assert subject.available(10.51, 3.)[0] is None


def test_inference_delay_does_not_add_future_publication_to_older_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D, 'perf_counter', lambda: 10.3)
    subject = owner([completed()])
    assert subject.available(10.3, 3.3, admitted_at=10.1)[0] is None
    assert subject.available(10.3, 3.3, admitted_at=10.2)[0] is not None


@pytest.mark.parametrize(('age', 'accepted'), [(.5, True), (.500001, False)])
def test_media_age_gate_independent_of_wall_clock(monkeypatch: pytest.MonkeyPatch,
                                                age: float, accepted: bool) -> None:
    monkeypatch.setattr(D, 'perf_counter', lambda: 10.3)
    subject = owner([completed()])
    assert (subject.available(10.3, 3. + age)[0] is not None) == accepted
