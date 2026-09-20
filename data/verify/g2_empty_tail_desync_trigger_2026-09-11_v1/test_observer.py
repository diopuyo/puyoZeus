"""原Adapter/実Invocationを使用。画像とmotionは人工で実動画認証ではない。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest
import observer as V

ORIGINAL = Path(__file__).resolve().parents[1] / 'g2_directional_next_enqueue_2026-09-09_v2'
sys.path.insert(0, str(ORIGINAL))
spec = importlib.util.spec_from_file_location('_desync_original_harness', ORIGINAL / 'test_occurrence.py')
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)
modules, h = T.modules, T.h
F = T.FRAME
BB, GY, YP, PP = (2, 2), (3, 4), (4, 5), (5, 5)


def mismatch(h: Any, side: str = '1P') -> V.Observer:
    observer = V.install(h.stack, h.adapter, side)
    h.drive(F, BB, GY, quiet=True, side=side)
    h.drive(F + 2, GY, YP, quiet=True, side=side)
    candidate = T.event(F + 4)
    h.drive(F + 4, YP, PP, event=candidate, pulse=True, side=side)
    h.provider.change = {'quiet_frames': None}
    h.drive(F + 6, YP, PP, quiet=True, side=side)
    assert observer.take(h.pipe, F + 6) is None
    h.provider.change = {}
    h.drive(F + 8, YP, PP, quiet=True, side=side)
    assert observer.take(h.pipe, F + 8) is None
    h.drive(F + 10, YP, PP, quiet=True, side=side)
    return observer


@pytest.mark.parametrize('side', ['1P', '2P'])
def test_actual_completed_update_advisory_once(h: Any, side: str) -> None:
    observer = mismatch(h, side)
    value = observer.take(h.pipe, F + 10)
    assert [f.frame for f in value.facts] == [F + 8, F + 10]
    assert value.invocation.runtime.pipe is h.pipe
    assert value.missing_count == 'UNCERTIFIED'
    assert not value.reset_permission and not value.current_permission
    assert not getattr(h.pipe, '_pending_tsumo_' + side.lower())
    assert observer.take(h.pipe, F + 10) is None
    h.drive(F + 12, YP, PP, quiet=True, side=side)
    assert observer.take(h.pipe, F + 12) is None
    T.invariant(h, side)


@pytest.mark.parametrize('case', ['pipe', 'frame', 'active', 'completed', 'state', 'epoch', 'error'])
def test_reject_false_completed_boundary(h: Any, case: str) -> None:
    observer = mismatch(h)
    value = observer.pending
    runtime = value.invocation.runtime
    pipe, frame = h.pipe, F + 10
    if case == 'pipe': pipe = object()
    if case == 'frame': frame += 2
    if case == 'active': h.controller.active = value.invocation
    if case == 'completed': runtime.completed -= 2
    if case == 'state': h.adapter.states[(id(pipe), '1P')] = dict(value.state)
    if case == 'epoch': runtime.histories['1P'] = h.N.History(1, None, frame, ())
    if case == 'error': value.invocation.error = ValueError('人工失敗')
    try:
        with pytest.raises(ValueError, match='desync_observer:'):
            observer.take(pipe, frame)
        assert observer.pending is value
    finally:
        h.controller.active = None


def test_normal_successor_not_reset_signal(h: Any) -> None:
    observer = V.install(h.stack, h.adapter)
    h.drive(F, BB, GY, quiet=True)
    h.drive(F + 2, GY, YP, event=T.event(F + 2), pulse=True)
    h.drive(F + 4, GY, YP, quiet=True)
    assert list(h.pipe._pending_tsumo_1p) == [BB]
    assert observer.take(h.pipe, F + 4) is None
    T.invariant(h)


def test_real_nonempty_queue_blocks_signal(h: Any) -> None:
    observer = V.install(h.stack, h.adapter)
    h.drive(F, BB, GY, quiet=True)
    h.drive(F + 2, GY, YP, event=T.event(F + 2), pulse=True)
    h.drive(F + 4, GY, YP, quiet=True)
    h.drive(F + 6, GY, YP, quiet=True)
    h.drive(F + 8, PP, BB, event=T.event(F + 8, sequence=2), pulse=True)
    h.drive(F + 10, PP, BB, quiet=True)
    h.drive(F + 12, PP, BB, quiet=True)
    assert observer.take(h.pipe, F + 12) is None
    assert list(h.pipe._pending_tsumo_1p) == [BB]
    T.invariant(h)
