"""画像コピー・同caller結合の人工試験。物理検出精度は測らない。"""
from __future__ import annotations

import ast
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import provider as P
from test_stream import C, SOURCE, dto, make, packet


class Controller:
    def __init__(self) -> None:
        self.active: Any = None
        self.main_calls = 0

    def _invocation(self, pipe: Any, frame: int, time_sec: float) -> Any:
        assert self.active.runtime.pipe is pipe
        assert self.active.frame == frame and self.active.time_sec == time_sec
        return self.active

    def main_return(self, pipe: Any, frame: int, time_sec: float, value: Any) -> Any:
        invocation = self._invocation(pipe, frame, time_sec)
        invocation.main = value
        self.main_calls += 1
        return value


def setup() -> tuple[Controller, P.Provider, Any]:
    controller = Controller()
    types = SimpleNamespace(MotionObservation=dto, MotionCandidate=dto)
    value = P.Provider(controller, types, SOURCE, 'synthetic_provider_test')
    runtime = SimpleNamespace(pipe=object(), histories={s: SimpleNamespace(epoch=0) for s in P.SIDES})
    return controller, value, runtime


def begin(controller: Controller, runtime: Any, frame: int, *, main: Any = True) -> Any:
    value = SimpleNamespace(runtime=runtime, frame=frame, time_sec=frame / C.FPS, main=main)
    controller.active = value
    return value


def measure(before: Any, after: Any, side: str, slot: str, rois: Any) -> dict[str, Any]:
    artificial = packet(make(), 100)['slots'][slot]
    # fixtureの点だけ。画像から算出した点群とは呼ばない。
    points = artificial['points']
    if side == '2P':
        shift = rois[slot][2] - C.ROIS['1P'][C.SLOTS.index(slot)][2]
        for point in points:
            for key in ('start', 'end', 'back'):
                point[key][0] += shift
    return {'group': before.group, 'side': side, 'slot': slot,
        'frame_before': before.frame, 'frame_after': after.frame,
        'time_before': before.frame / C.FPS, 'time_after': after.frame / C.FPS,
        'png_before_sha256': before.sha256, 'png_after_sha256': after.sha256,
        'roi_y1_y2_x1_x2': list(rois[slot]), 'summary': {'seed_count': len(points)}, 'points': points}


def test_image_copy_and_actual_invocation_identity(monkeypatch: Any) -> None:
    controller, value, runtime = setup()
    monkeypatch.setattr(value.o, 'measure_slot', measure)
    pixels = np.zeros((720, 1280, 3), dtype=np.uint8)
    first = begin(controller, runtime, 100)
    value.capture(first, pixels, True)
    old = value.previous[(id(runtime.pipe), '1P')].gray
    pixels[:] = 255
    assert not old.any() and not old.flags.writeable
    assert value(first, '1P').invocation is first
    for frame in (102, 104):
        invocation = begin(controller, runtime, frame)
        value.capture(invocation, pixels, True)
    assert value(invocation, '1P').quiet_frames == (102, 104)
    with pytest.raises(ValueError, match='duplicate'):
        value.capture(invocation, pixels, True)


def test_inactive_missing_gap_and_epoch_reset(monkeypatch: Any) -> None:
    controller, value, runtime = setup()
    monkeypatch.setattr(value.o, 'measure_slot', measure)
    pixels = np.zeros((720, 1280, 3), dtype=np.uint8)
    for frame in (100, 102, 104):
        invocation = begin(controller, runtime, frame)
        value.capture(invocation, pixels, True)
    previous = value(invocation, '1P').segment_id
    invocation = begin(controller, runtime, 108)
    value.capture(invocation, pixels, True)
    assert value(invocation, '1P').segment_id != previous
    assert value(invocation, '1P').quiet_frames is None
    invocation = begin(controller, runtime, 110)
    assert value(invocation, '1P') is None
    runtime.histories['1P'].epoch += 1
    invocation = begin(controller, runtime, 112)
    value.capture(invocation, pixels, True)
    assert value(invocation, '1P').software_epoch == 1
    invocation = begin(controller, runtime, 114)
    value.capture(invocation, pixels, False)
    assert value(invocation, '1P') is None


def caller(controller: Any) -> Any:
    source = 'def update(self, frame_idx, time_sec, frame, is_active, result):\n' \
             '    return __next_live.main_return(self, frame_idx, time_sec, result)\n'
    tree = ast.increment_lineno(ast.parse(source), P.MAIN_NEXT_LINE - 2)
    namespace = {'__next_live': controller}
    exec(compile(tree, '<synthetic_update_caller_not_original_runtime>', 'exec'), namespace)
    return namespace['update']


def test_same_caller_return_once_and_restore() -> None:
    controller, value, runtime = setup()
    update = caller(controller)
    pixels, result = np.zeros((720, 1280, 3), dtype=np.uint8), object()
    invocation = begin(controller, runtime, 100, main=None)
    with ExitStack() as stack:
        P.attach(stack, controller, value, update.__code__)
        assert update(runtime.pipe, 100, 100 / C.FPS, pixels, True, result) is result
        assert controller.main_calls == 1 and value.captured is invocation
        with pytest.raises(ValueError, match='unexpected_image_caller'):
            controller.main_return(runtime.pipe, 100, 100 / C.FPS, result)
        assert controller.main_calls == 1
    assert 'main_return' not in vars(controller)


def test_bad_image_fault_is_sticky() -> None:
    controller, value, runtime = setup()
    invocation = begin(controller, runtime, 100)
    with pytest.raises(ValueError, match='image_dtype'):
        value.capture(invocation, np.zeros((720, 1280, 3), dtype=float), True)
    assert value.error is not None
    with pytest.raises(ValueError, match='provider_fault'):
        value(invocation, '1P')
