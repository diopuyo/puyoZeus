"""更新入口の輸送契約。leaseはstub、実resetは別の原105連続検証で行う。"""
from __future__ import annotations
from contextlib import ExitStack
from types import SimpleNamespace as N
from typing import Any
import pytest
import entry_boundary as E

FRAME = 34932


class Pipe:
    def __init__(self) -> None:
        self.order: list[str] = []
        self.result = object()

    def update(self, frame_idx: int, time_sec: float, frame: Any) -> Any:
        self.order.append('update')
        return self.result


def sample() -> tuple[Any, Any, Any]:
    pipe = Pipe()
    rows = [dict(frame_idx=FRAME-2, side=s, time_sec=(FRAME-2)/60, returned=None) for s in E.SIDES]
    sink = N(rows=rows, busy=False, closed=False, errors=[])
    def perform(recovery: Any, frame: int, clock: float) -> None:
        assert frame == FRAME and clock == FRAME/60
        pipe.order.append('reset')
    return pipe, dict(collector_metadata_sink=sink), N(perform=perform)


def test_one_reset_between_metadata_and_original_update() -> None:
    pipe, state, lease = sample()
    with ExitStack() as stack:
        entry = E.install(stack, pipe, state, lease, object(), FRAME)
        assert pipe.update(frame_idx=FRAME, time_sec=FRAME/60, frame=None) is pipe.result
        assert pipe.update(FRAME+2, (FRAME+2)/60, None) is pipe.result
        assert pipe.order == ['reset', 'update', 'update'] and entry.done
    assert 'update' not in vars(pipe)


@pytest.mark.parametrize('case', ['busy', 'closed', 'errors', 'missing', 'clock', 'frame', 'side', 'returned'])
def test_incomplete_previous_metadata_never_calls_reset_or_update(case: str) -> None:
    pipe, state, lease = sample()
    sink = state['collector_metadata_sink']
    if case in ('busy', 'closed'): setattr(sink, case, True)
    if case == 'errors': sink.errors.append('人工失敗')
    if case == 'missing': sink.rows.pop()
    if case == 'clock': sink.rows[-1]['time_sec'] += 1
    if case == 'frame': sink.rows[-1]['frame_idx'] -= 2
    if case == 'side': sink.rows[-1]['side'] = '1P'
    if case == 'returned': sink.rows[-1].pop('returned')
    with ExitStack() as stack:
        entry = E.install(stack, pipe, state, lease, object(), FRAME)
        with pytest.raises(ValueError, match='reset_entry:metadata_'):
            pipe.update(FRAME, FRAME/60, None)
        assert not entry.done and not pipe.order


@pytest.mark.parametrize('case', ['frame', 'clock', 'lease'])
def test_boundary_failure_is_sticky(case: str) -> None:
    pipe, state, lease = sample()
    if case == 'lease':
        def fail(*args: Any) -> None: raise RuntimeError('人工lease失敗')
        lease.perform = fail
    frame, clock = FRAME+(2 if case == 'frame' else 0), FRAME/60+(1 if case == 'clock' else 0)
    with ExitStack() as stack:
        entry = E.install(stack, pipe, state, lease, object(), FRAME)
        with pytest.raises((ValueError, RuntimeError)) as first: pipe.update(frame, clock, None)
        with pytest.raises(type(first.value)) as second: pipe.update(FRAME, FRAME/60, None)
        assert second.value is first.value and entry.error is first.value and not pipe.order
