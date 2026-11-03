"""入力切替を監視し、機器別ウォームアップが終わるまで認識通知を配信しない。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Callable, Iterator

from src.production_config import FAST_FIRST_BACKGROUND_RECHECK
from . import live_verify_split as split
from .live_calibration import ColorWarmup
from .live_device import DeviceConfig, DirectShowSource, VERIFY_PERIOD_SEC
from .live_source import CapturedFrame, FrameSource, RECOGNITION_HZ


class InputChanged(Exception):
    """captureを解放して新しい設定へ切り替える内部制御。"""


class ConfiguredSource(FrameSource):
    def __init__(self, path: Path, duration: float, on_switch: Callable, on_status: Callable,
                 source_factory: Callable = DirectShowSource,
                 on_format_change: Callable[[], None] | None = None,
                 on_diagnostics: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.path, self.duration = path, duration
        self.on_switch, self.on_status, self.source_factory = on_switch, on_status, source_factory
        self.last_check = float('-inf')
        self.dropped = 0
        self.on_format_change = on_format_change
        self.on_diagnostics = on_diagnostics
        self.on_match_end_hold: Callable[[Any, float, float], None] | None = None

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
                if self.on_diagnostics:
                    self.on_diagnostics(dict(getattr(source, 'input_diagnostics', {})))
                self.on_status(phase)
            source = self.source_factory(config, self.duration-(time.perf_counter()-origin),
                                         lambda: hold('verifying'), on_status=hold)
            if self.on_match_end_hold is not None:
                source.on_match_end_hold = lambda image, stamp: self.on_match_end_hold(image, stamp-origin, stamp)
            if config.fast_first_judgment:
                source.on_format_change = self.on_format_change
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

    # live_verify_soft_hold (既定 OFF)。クラス属性の既定にして、属性を手動設定する試験用の構築でも従来どおり動かす。
    # (__getattr__ が未設定の名前を pipe へ委譲するので、getattr の既定値には頼らない)
    _soft_enabled = False
    _soft_hold: split.SoftHoldTimer | None = None
    _in_match = False

    def __init__(self, pipe: Any, path: Path, duration: float, queue: Any) -> None:
        self.pipe, self.queue = pipe, queue
        self.base_ranges = deepcopy(pipe._reader._classifier._hsv._ranges)
        self.calibrator_template = deepcopy(pipe._online_hsv)
        self.pipe._online_hsv = None
        self.warmup: ColorWarmup | None = None
        self.epoch = -1
        self.publishing_ready = False
        self.last_status: dict | None = None
        self._held = False
        self._calibration_invalid = False
        self._recheck: ColorWarmup | None = None
        self._needs_recheck = False
        self._input_diagnostics: dict[str, Any] = {}
        self._soft_enabled = split.enabled()
        self._soft_hold = split.SoftHoldTimer() if self._soft_enabled else None
        self.source = ConfiguredSource(path, duration, self.switch, self.hold,
                                       on_format_change=self.invalidate_format,
                                       on_diagnostics=self.remember_input_diagnostics)
        from .live_match_end_feed import enabled, feed
        if enabled() and getattr(pipe, '_enable_match_end_pair', False):
            self.source.on_match_end_hold = lambda image, stamp, captured: feed(self, image, stamp, captured)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.pipe, name)

    def switch(self, config: DeviceConfig, advance_epoch: bool = True) -> None:
        self._match_end_hold_eligible = False
        self._held, self._calibration_invalid, self._recheck = False, False, None
        self._needs_recheck = False
        self.epoch += int(advance_epoch)
        self.pipe.reset()
        self._forget_match()
        hsv = self.pipe._reader._classifier._hsv
        hsv._ranges = deepcopy(self.base_ranges)
        hsv._native_ranges_cache = hsv._native_params_cache = None
        self.warmup = ColorWarmup(config, verification_only=config.verification_only,
                                  calibrator=self.calibrator_template)
        self.publishing_ready = False
        self.emit(self.warmup.status('verifying'))

    def emit(self, status: dict) -> None:
        status = dict(status, epoch=self.epoch)
        alignment = getattr(self.pipe, '_score_roi_alignment', None)
        if alignment is not None:
            status['score_roi_alignment'] = alignment.status()
        if status != self.last_status:
            # 比較対象へ計装を混ぜず、従来と同じ状態遷移だけを送る。
            self.queue.put(('hold', dict(status, at=time.perf_counter(),
                input_diagnostics=dict(self._input_diagnostics) if status['phase'] != 'ready' else None)))
            self.last_status = status

    def remember_input_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        """入力保留の原因を次の状態通知に添える。判断には使用しない。"""
        self._input_diagnostics = diagnostics

    def hold(self, phase: str) -> None:
        if getattr(self, '_match_end_hold_eligible', False):
            self._clear_soft_hold()   # eligible 中は期限を進めない (1fd4b1cf の無期限の抑止はそのまま)
            self.publishing_ready = False
            self.emit(self.warmup.status(phase) if self.warmup else dict(phase=phase, progress=0))
            return
        if self._keeps_state_on_hold():
            self.publishing_ready = False
            self.emit(self.warmup.status(phase))
            return
        if (self.warmup is not None and getattr(self.warmup.device, 'fast_first_judgment', False)
                and not self._calibration_invalid):
            self.hold_retaining_calibration(phase)
            return
        if self.warmup is not None and (self.warmup.ready or self.warmup.progress > 0):
            self.switch(self.warmup.device, advance_epoch=not self._held)
        self.publishing_ready = False
        self.emit(self.warmup.status(phase) if self.warmup else dict(phase=phase, progress=0))

    def _keeps_state_on_hold(self) -> bool:
        """(b) 一時的な見え方の不足を、試合中・期限内に限り、状態を捨てずに保留する (live_verify_soft_hold)。

        次のどれかなら従来どおり (reset・epoch+1) へ進む: フラグ OFF・従来も reset しない保留・
        試合中でない・(a) 入力の異常・期限超過。判定は保留のたびにやり直す (保留中に (a) へ変われば即 reset)。
        """
        if not self._soft_enabled or self.warmup is None:
            return False
        if not (self.warmup.ready or self.warmup.progress > 0):
            return False
        transient = self._input_diagnostics.get(split.KIND_KEY) == split.KIND_TRANSIENT
        if not self._in_match or not transient:
            self._clear_soft_hold()
            return False
        return self._soft_hold.within_limit()

    def _clear_soft_hold(self) -> None:
        """保留の期限の起点を消す (復帰・eligible・reset のとき)。タイマーが無い (OFF) なら何もしない。"""
        if self._soft_hold is not None:
            self._soft_hold.clear()

    def _forget_match(self) -> None:
        """pipe.reset() の直後に呼ぶ。soft hold が使う「試合中」の記録と期限を消す (OFF は何もしない)。"""
        if self._soft_enabled:
            self._in_match = False
            self._clear_soft_hold()

    def invalidate_format(self) -> None:
        """解像度変更は、次の既存holdで較正も初期化する。"""
        self._calibration_invalid = True

    def hold_retaining_calibration(self, phase: str) -> None:
        """保留の初回だけ従来の世代更新・履歴リセットを行い、較正だけ残す。"""
        if not self._held and (self.warmup.ready or self.warmup.progress > 0):
            self.epoch += 1
            self.pipe.reset()
            self._forget_match()
            self._recheck = None
            self._needs_recheck = self.warmup.ready
            self.emit(self.warmup.status('verifying'))
            self._held = True
        self.publishing_ready = False
        self.emit(self.warmup.status(phase))

    def recheck_calibration(self, image: Any, result: Any) -> bool:
        """既存の保存profile照合を裏で継続し、色ずれ時だけ全サンプル再収集へ戻す。"""
        if not self._needs_recheck or not self.warmup.ready:
            return False
        if self._recheck is None:
            self._recheck = ColorWarmup(self.warmup.device, self.warmup.verification_only,
                                        self.calibrator_template)
        self._recheck.observe(image, result, self.pipe._reader._classifier)
        if (FAST_FIRST_BACKGROUND_RECHECK and self.warmup.verification_only
                and self.warmup.device.fast_first_judgment):
            # 検証専用では新旧どちらの範囲も認識器へ適用しない。履歴は壊さず収集する。
            if self._recheck.ready:
                self.warmup = self._recheck
                self._recheck, self._needs_recheck = None, False
            return False
        if self._recheck.changed_profile:
            recheck = self._recheck
            # 色ずれは入力世代の変更ではない。既存の照合失敗と同じく強制境界を増やさない。
            self.pipe.reset()
            self._forget_match()
            hsv = self.pipe._reader._classifier._hsv
            hsv._ranges = deepcopy(self.base_ranges)
            hsv._native_ranges_cache = hsv._native_params_cache = None
            self.publishing_ready = False
            self._recheck, self._needs_recheck = None, False
            self.warmup = recheck
            if recheck.ready and not recheck.verification_only:
                # 同じ観測で全量収集まで済んだ場合も、初期値へ戻した補正を再適用する。
                recheck.advance(self.pipe._reader._classifier)
            self.emit(self.warmup.status())
            return True
        elif self._recheck.ready:
            self._recheck = None
            self._needs_recheck = False
        return False

    def update(self, index: int, stamp: float, image: Any) -> Any:
        self._match_end_hold_eligible = False
        self._held = False
        self.restore_calibration_metadata()
        result = self.pipe.update(index, stamp, image)
        from src.row1_publication import evaluation_result
        internal = evaluation_result(result)
        if self._soft_enabled:
            # 復帰した。次の保留は新しい起点から数える。試合中かは直近の認識結果で決める。
            self._in_match = bool(getattr(internal, 'is_match_active', False))
            self._clear_soft_hold()
        if getattr(getattr(self.warmup, 'device', None), 'fast_first_judgment', False):
            if self.recheck_calibration(image, internal):
                return result
        previously_ready = self.warmup.ready
        status = self.warmup.observe(image, internal, self.pipe._reader._classifier)
        self.publishing_ready = self.warmup.ready
        if self.warmup.ready and not previously_ready and not self.warmup.verification_only:
            # 補正前の確定盤面を公開せず、補正後の画面から履歴を作り直す。
            self.pipe.reset()
            self._forget_match()
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
