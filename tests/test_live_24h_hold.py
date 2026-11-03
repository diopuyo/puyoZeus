"""L200Hの保持期限・解除・側別欠測を検査する。"""
import math

import pytest

from src.phase_j.live_ex_hold import LiveExHold
from src.phase_j._live_ex_frozen.src.dl1k_columns import COLUMNS, LEARNING_COLUMNS


def source() -> dict:
    extra = dict.fromkeys(COLUMNS, 1.)
    extra.update(av_ex_p1=.2, av_ex_p2=.2)
    return dict(extra=extra, input_t=0., eligible_at=10.2)


def event(t: float, **changes: object) -> dict:
    return dict(dict(group=(1, 1), t=t, row_key=str(t), fight=True,
                     reset={'p1': False, 'p2': False}), **changes)


@pytest.mark.parametrize('reason', ['not_yet', 'stale', 'not_completed', 'key_mismatch'])
def test_transport_hold_does_not_renew(reason: str) -> None:
    hold = LiveExHold()
    hold.adopt(event(.2), source(), 'ok')
    assert hold.adopt(event(.7), None, reason)[LEARNING_COLUMNS[0]] == 1.
    assert math.isnan(hold.adopt(event(.700001), None, reason)[LEARNING_COLUMNS[0]])


def test_reset_one_side_and_boundary() -> None:
    hold = LiveExHold()
    hold.adopt(event(.2), source(), 'ok')
    values = hold.adopt(event(.3, reset={'p1': True, 'p2': False}), None, 'stale')
    assert all(math.isnan(values[c]) for c in LEARNING_COLUMNS if c.endswith('_p1'))
    assert all(values[c] == 1. for c in LEARNING_COLUMNS if c.endswith('_p2'))
    values = hold.adopt(event(.4, group=(1, 2)), None, 'stale')
    assert all(math.isnan(values[c]) for c in LEARNING_COLUMNS)


def test_revalidation_does_not_reattach_and_reset_invalidates() -> None:
    hold, key = LiveExHold(), dict(key=(1, 1, (0., None)), fight=True)
    raw, _ = hold.available(10.2, .2, 10.2, key, source(), 'available', 1)
    assert raw is not None
    raw, _ = hold.available(10.51, .2, 10.2, key, None, 'not_ready_or_stale', 1)
    assert raw is None
    raw, why = hold.available(10.6, .6, 10.6, key, None, 'not_ready_or_stale', 1)
    assert raw is not None and why == 'held'
    hold.observe(dict(key, resets=[dict(side='p1', event='fire')]))
    raw, _ = hold.available(10.61, .6, 10.6, key, None, 'not_ready_or_stale', 1)
    assert all(math.isnan(raw['extra'][c]) for c in LEARNING_COLUMNS if c.endswith('_p1'))


def test_non_transport_failure_clears() -> None:
    hold = LiveExHold()
    hold.adopt(event(.2), source(), 'ok')
    values = hold.adopt(event(.3), None, 'worker_failed')
    assert all(math.isnan(values[c]) for c in LEARNING_COLUMNS)
