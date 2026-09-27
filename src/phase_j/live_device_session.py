"""入力切替を監視し、機器別ウォームアップが終わるまで認識通知を配信しない。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Callable, Iterator

from .live_calibration import ColorWarmup
from .live_device import DeviceConfig, DirectShowSource, VERIFY_PERIOD_SEC
from .live_source import CapturedFrame, FrameSource, RECOGNITION_HZ


class InputChanged(Exception):
    """captureを解放して新しい設定へ切り替える内部制御。"""


class ConfiguredSource(FrameSource):
    def __init__(self, path: Path, duration: float, on_switch: Callable, on_status: Callable,
                 source_factory: Callable = DirectShowSource) -> None:
        self.path, self.duration = path, duration
        self.on_switch, self.on_status, self.source_factory = on_switch, on_status, source_factory
        self.last_check = float('-inf')
        self.dropped = 0

    def check(self, config: DeviceConfig) -> None:
        if time.perf_counter()-self.last_check < VERIFY_PERIOD_SEC:
            return
        self.last_check = time.perf_counter()
        try:
            if DeviceConfig.load(self.path) == config:
                return
        except (OSError, ValueError, KeyError, TypeError):
            pass
        raise InputChanged()

    def __iter__(self) -> Iterator[CapturedFrame]:
        origin = time.perf_counter()
        while time.perf_counter()-origin < self.duration:
            try:
                config = DeviceConfig.load(self.path)
            except (OSError, ValueError, KeyError, TypeError):
                self.on_status('verifying')
                time.sleep(VERIFY_PERIOD_SEC)
                continue
            self.on_switch(config)
            def hold(phase: str) -> None:
                self.check(config)
                self.on_status(phase)
            source = self.source_factory(config, self.duration-(time.perf_counter()-origin),
                                         lambda: hold('verifying'), on_status=hold)
            iterator = iter(source)
            try:
                for frame in iterator:
                    self.check(config)
                    elapsed = frame.captured_at-origin
                    yield replace(frame, index=round(elapsed*RECOGNITION_HZ), media_sec=elapsed)
            except InputChanged:
                self.on_status('verifying')
                continue
            finally:
                iterator.close()
            break


class DeviceSession:
    """認識器のproxy。機器切替・入力喪失後に履歴と較正を再初期化する。"""

    def __init__(self, pipe: Any, path: Path, duration: float, queue: Any) -> None:
        self.pipe, self.queue = pipe, queue
        self.base_ranges = deepcopy(pipe._reader._classifier._hsv._ranges)
        self.calibrator_template = deepcopy(pipe._online_hsv)
        self.pipe._online_hsv = None
        self.warmup: ColorWarmup | None = None
        self.epoch = -1
        self.publishing_ready = False
        self.last_status: dict | None = None
        self.source = ConfiguredSource(path, duration, self.switch, self.hold)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.pipe, name)

    def switch(self, config: DeviceConfig) -> None:
        self.epoch += 1
        self.pipe.reset()
        hsv = self.pipe._reader._classifier._hsv
        hsv._ranges = deepcopy(self.base_ranges)
        hsv._native_ranges_cache = hsv._native_params_cache = None
        self.warmup = ColorWarmup(config, verification_only=config.verification_only,
                                  calibrator=self.calibrator_template)
        self.publishing_ready = False
        self.emit(self.warmup.status('verifying'))

    def emit(self, status: dict) -> None:
        status = dict(status, epoch=self.epoch)
        if status != self.last_status:
            self.queue.put(('hold', dict(status, at=time.perf_counter())))
            self.last_status = status

    def hold(self, phase: str) -> None:
        if self.warmup is not None and (self.warmup.ready or self.warmup.progress > 0):
            self.switch(self.warmup.device)
        self.publishing_ready = False
        self.emit(self.warmup.status(phase) if self.warmup else dict(phase=phase, progress=0))

    def update(self, index: int, stamp: float, image: Any) -> Any:
        self.restore_calibration_metadata()
        result = self.pipe.update(index, stamp, image)
        previously_ready = self.warmup.ready
        status = self.warmup.observe(image, result, self.pipe._reader._classifier)
        self.publishing_ready = self.warmup.ready
        if self.warmup.ready and not previously_ready and not self.warmup.verification_only:
            # 補正前の確定盤面を公開せず、補正後の画面から履歴を作り直す。
            self.pipe.reset()
            self.publishing_ready = False
        self.restore_calibration_metadata()
        self.emit(status)
        return result

    def restore_calibration_metadata(self) -> None:
        """較正済み色を既存のresyncガードへ通知する。試合reset後も維持する。"""
        if self.warmup and self.warmup.ready:
            self.pipe._online_hsv_injected = True
            self.pipe._online_hsv_injected_colors = {
                c for c in (1, 2, 3, 4, 5) if self.warmup.calibrator.is_ready(c)}
