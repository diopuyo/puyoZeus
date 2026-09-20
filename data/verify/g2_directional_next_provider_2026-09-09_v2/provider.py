"""同update画像から実点群を測り、同invocationの限定票を返す。"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import dependencies as D
from stream import Stream

SIDES = ('1P', '2P')
MAIN_NEXT_LINE = 5061


class Provider:
    def __init__(self, controller: Any, types: Any, source: str, run_id: str) -> None:
        self.controller, self.types, self.source, self.run_id = controller, types, source, run_id
        self.c, self.o, self.p = D.load('candidate'), D.load('observables'), D.load('packet')
        self.c.require(type(run_id) is str and bool(run_id), 'run_id')
        self.o.configure()
        self.rois = self.o.read_rois()
        self.streams: dict[tuple[int, str], Stream] = {}
        self.previous: dict[tuple[int, str], Any] = {}
        self.captured: Any = None
        self.error: str | None = None
        self.records: list[dict[str, Any]] = []

    def stream(self, invocation: Any, side: str) -> Stream:
        key = (id(invocation.runtime.pipe), side)
        if key not in self.streams:
            self.streams[key] = Stream(self.c, self.types, self.source, side)
        return self.streams[key]

    def gray(self, pixels: Any) -> Any:
        o = self.o
        o.require(type(pixels) is o.np.ndarray and pixels.dtype == o.np.uint8, 'image_dtype')
        o.require(pixels.shape in ((720, 1280, 3), (1080, 1920, 3)), 'image_shape')
        color = o.cv2.resize(pixels, o.TARGET_SIZE, interpolation=o.cv2.INTER_AREA) \
            if pixels.shape[:2] != (o.TARGET_SIZE[1], o.TARGET_SIZE[0]) else pixels
        gray = o.cv2.cvtColor(color, o.cv2.COLOR_BGR2GRAY)
        gray.flags.writeable = False
        return gray

    def side(self, invocation: Any, side: str, gray: Any, digest: str, active: bool) -> None:
        key, stream = (id(invocation.runtime.pipe), side), self.stream(invocation, side)
        epoch = invocation.runtime.histories[side].epoch
        old = self.previous.get(key)
        if not active:
            stream.invalidate()
            self.previous.pop(key, None)
            return
        if stream.epoch != epoch or old is None or invocation.frame - old.frame != self.c.STRIDE:
            stream.invalidate()
            stream.start(invocation, epoch)
            self.previous[key] = self.o.SavedImage(stream.sequence(), invocation.frame, digest, gray)
            return
        current = self.o.SavedImage(stream.sequence(), invocation.frame, digest, gray)
        rows = [self.o.measure_slot(old, current, side, slot, self.rois[side]) for slot in self.o.SLOTS]
        packet = self.p.packet(self.c, rows, self.source)
        result = stream.update(invocation, packet)
        self.previous[key] = current
        self.records.append({'side': side, 'frame': invocation.frame, 'epoch': epoch,
            'segment_id': result.segment_id, 'quiet_frames': result.quiet_frames,
            'candidate': vars(result.candidate) if result.candidate is not None else None,
            'gray_before_sha256': old.sha256, 'gray_after_sha256': digest,
            'observed': stream.last_record, 'physical_progress_certified': False})

    def capture(self, invocation: Any, pixels: Any, active: bool) -> None:
        self.c.require(self.error is None and self.captured is not invocation, 'duplicate_or_faulted_capture')
        self.c.require(type(active) is bool and invocation.main is not None, 'active_or_main')
        try:
            gray = self.gray(pixels)
            digest = self.o.sha(gray.tobytes())
            for side in SIDES:
                self.side(invocation, side, gray, digest, active)
            self.captured = invocation
        except BaseException as exc:
            self.error = repr(exc)
            for side in SIDES:
                self.stream(invocation, side).invalidate()
            raise

    def __call__(self, invocation: Any, side: str) -> Any:
        self.c.require(side in SIDES and self.error is None, 'side_or_provider_fault')
        actual = self.controller._invocation(invocation.runtime.pipe, invocation.frame, invocation.time_sec)
        self.c.require(actual is invocation, 'provider_invocation_identity')
        stream = self.stream(invocation, side)
        if self.captured is not invocation:
            stream.invalidate()
            self.previous.pop((id(invocation.runtime.pipe), side), None)
            return None
        self.c.require(stream.latest is None or stream.latest.invocation is invocation, 'stale_observation')
        return stream.latest


def attach(stack: Any, controller: Any, provider: Provider, expected_code: Any) -> None:
    original = controller.main_return
    provider.c.require('main_return' not in vars(controller), 'already_wrapped_main_return')
    def main_return(pipe: Any, frame: int, time_sec: float, result: Any) -> Any:
        caller = sys._getframe(1)
        provider.c.require(caller.f_code is expected_code and caller.f_lineno == MAIN_NEXT_LINE,
                           'unexpected_image_caller')
        provider.c.require(caller.f_globals.get('__next_live') is controller, 'caller_controller')
        local = caller.f_locals
        provider.c.require(local.get('self') is pipe and local.get('frame_idx') == frame
                           and local.get('time_sec') == time_sec, 'caller_clock')
        returned = original(pipe, frame, time_sec, result)
        invocation = controller._invocation(pipe, frame, time_sec)
        provider.capture(invocation, local.get('frame'), local.get('is_active'))
        return returned
    stack.callback(vars(controller).pop, 'main_return', None)
    controller.main_return = main_return
