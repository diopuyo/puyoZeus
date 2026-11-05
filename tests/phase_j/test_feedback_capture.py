"""合成画像だけで、仮想カメラから公開警告までを確認する。"""
from __future__ import annotations

import time
from pathlib import Path
from collections.abc import Iterator
from typing import Any
import cv2
import numpy as np
import pytest

from src.phase_j.distribution_input import has_feedback, marker_image
from src.phase_j.dshow_color import BT709_KR_KB, correct_601_as_709
from src.phase_j.live_device import DeviceConfig, DirectShowSource
from src.phase_j.live_publish import input_hold
from src.phase_j.live_speed import SPEED_DEFAULTS
from src.production_config import RELEASE_LIVE_DEFAULTS, RELEASE_LIVE_ENV_DEFAULTS
from tests.phase_j.test_dshow_color import _encode

SIZE = (1080, 1920, 3)
MARKER_Y, MARKER_X = 185, 948
STATIC_END, LOST_END, CLEAN_END, RUN_END = 1.2, 1.5, 1.8, 2.9
DETECTION_LIMIT_SEC = .5
RANDOM_SEED = 20261105
COMPARISON_SIZE = (60, 60, 3)


@pytest.fixture(autouse=True)
def release_opencv_threads() -> Iterator[None]:
    """配布launcherのcpu_threads=1を使い、他の試験へ設定を残さない。"""
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(previous)


def encoded_camera(image: np.ndarray) -> np.ndarray:
    """既存色変換試験の独立した参照式で709/I420→601を作る。"""
    ycc = _encode(image[..., ::-1], BT709_KR_KB)
    height, width = image.shape[:2]
    planes = [ycc[..., 0]] + [cv2.resize(ycc[..., channel], (width // 2, height // 2),
        interpolation=cv2.INTER_AREA) for channel in (1, 2)]
    packed = np.concatenate([np.clip(np.rint(p), 0, 255).astype(np.uint8).ravel() for p in planes])
    return cv2.cvtColor(packed.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_I420)


def marked_image(dx: int = 0, dy: int = 0, scale: float = 1.0) -> np.ndarray:
    image = np.full(SIZE, 32, dtype=np.uint8)
    marker = cv2.resize(marker_image(), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    height, width = marker.shape[:2]
    y, x = MARKER_Y + dy, MARKER_X + dx
    image[y:y + height, x:x + width] = marker
    return image


def release_environment(monkeypatch: pytest.MonkeyPatch, buffered: bool, asynchronous: bool) -> None:
    for key, value in RELEASE_LIVE_ENV_DEFAULTS.items():
        monkeypatch.setenv(key, value)
    for key, value in {**SPEED_DEFAULTS, **RELEASE_LIVE_DEFAULTS}.items():
        if isinstance(value, bool):
            monkeypatch.setenv('PUYO_' + key.upper(), str(int(value)))
    monkeypatch.setenv('PUYO_LIVE_CAPTURE_THREAD', str(int(buffered)))
    monkeypatch.setenv('PUYO_LIVE_VERIFY_ASYNC', str(int(asynchronous)))
    monkeypatch.setenv('PUYO_LIVE_ASYNC_SCREEN_VERIFY', '0')


@pytest.mark.parametrize('dx,dy', [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize('scale', [1.0, .8, .7, .5])
def test_camera_marker_phases(dx: int, dy: int, scale: float) -> None:
    raw = encoded_camera(marked_image(dx, dy, scale))
    assert has_feedback(raw)
    assert has_feedback(correct_601_as_709(raw))


@pytest.mark.parametrize('y,x', [(0, 0), (0, 1896), (1062, 0), (1062, 1896)])
def test_camera_marker_at_image_edges(y: int, x: int) -> None:
    """狭めた探索範囲でも、画面四隅の目印全体を照合する。"""
    image = np.full(SIZE, 32, dtype=np.uint8)
    marker = marker_image()
    height, width = marker.shape[:2]
    image[y:y + height, x:x + width] = marker
    assert has_feedback(encoded_camera(image))


@pytest.mark.parametrize('color', [(0, 0, 0), (197, 55, 225), (255, 255, 255), (32, 64, 96)])
def test_solid_colors_are_not_feedback(color: tuple[int, int, int]) -> None:
    image = np.empty(SIZE, dtype=np.uint8)
    image[:] = color
    assert not has_feedback(image)
    assert not has_feedback(encoded_camera(image))


@pytest.mark.parametrize('noise', [0, 25, 80])
def test_pruning_equals_exhaustive_matching(noise: int) -> None:
    """相関上限による省略が、全参照を照合した結果と一致する。"""
    from src.phase_j.feedback_templates import templates
    from src.phase_j.distribution_input import _matches_marker, MARKER_TOLERANCE
    rng = np.random.default_rng(RANDOM_SEED)
    for template in templates():
        crop = np.full(COMPARISON_SIZE, 32, dtype=np.uint8)
        height, width = template.shape[:2]
        noisy = template.astype(float) + rng.normal(0, noise, template.shape)
        crop[:height, :width] = np.clip(noisy, 0, 255).astype(np.uint8)
        exhaustive = any(cv2.minMaxLoc(cv2.matchTemplate(crop, reference,
            cv2.TM_CCOEFF_NORMED))[1] >= MARKER_TOLERANCE for reference in templates())
        assert _matches_marker(crop) == exhaustive


class Camera:
    def __init__(self, image: np.ndarray, lifecycle: bool = True) -> None:
        self.image, self.lifecycle = image, lifecycle
        self.start, self.count, self.released = time.perf_counter(), 0, False

    def isOpened(self) -> bool:
        return not self.released

    def set(self, key: int, value: float) -> bool:
        return True

    def get(self, key: int) -> float:
        return 0.0

    def read(self) -> tuple[bool, Any]:
        elapsed = time.perf_counter() - self.start
        self.count += 1
        if self.lifecycle and STATIC_END <= elapsed < LOST_END:
            return False, None
        if self.lifecycle and LOST_END <= elapsed < CLEAN_END:
            image = np.full(SIZE, 32, dtype=np.uint8)
            image[:8, :8] = self.count % 255
            return True, image
        return True, self.image.copy()

    def release(self) -> None:
        self.released = True


@pytest.mark.parametrize('buffered', [False, True])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_static_loss_recovery(monkeypatch: pytest.MonkeyPatch, buffered: bool, asynchronous: bool) -> None:
    release_environment(monkeypatch, buffered, asynchronous)
    camera = Camera(encoded_camera(marked_image()))
    statuses: list[tuple[float, str]] = []
    source = DirectShowSource(DeviceConfig('OBS Virtual Camera', 0, color_correction='601to709'),
        RUN_END, lambda: None, lambda *args: camera, verifier=lambda image: True,
        on_status=lambda phase: statuses.append((time.perf_counter() - camera.start, phase)))
    frames = list(source)
    assert camera.released
    first = next(stamp for stamp, phase in statuses if phase == 'overlay_feedback')
    assert first < 3.0
    assert all(phase == 'overlay_feedback' for stamp, phase in statuses if first <= stamp < STATIC_END)
    assert any(phase == 'verifying' for stamp, phase in statuses if STATIC_END <= stamp < LOST_END)
    assert all(phase != 'overlay_feedback' for stamp, phase in statuses if LOST_END <= stamp < CLEAN_END)
    # 復帰後の再検査周期までは、前回のclean検査結果を使用する。
    assert all(frame.media_sec < first or frame.media_sec >= LOST_END for frame in frames)
    assert any(stamp >= CLEAN_END and phase == 'overlay_feedback' for stamp, phase in statuses)


def test_corrected_image_is_checked_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j import distribution_input
    source = DirectShowSource(DeviceConfig('synthetic', 0), 0, lambda: None)
    raw, corrected = np.zeros(SIZE, dtype=np.uint8), marked_image()
    seen = []
    def detect(image: np.ndarray) -> bool:
        seen.append(image)
        return bool(np.any(image))
    monkeypatch.setattr(distribution_input, 'has_feedback', detect)
    source._check_feedback(raw, 0, corrected)
    try:
        deadline = time.perf_counter() + 3.0
        while source._feedback_worker.result(0)[1] != 0 and time.perf_counter() < deadline:
            time.sleep(.01)
        assert source._feedback_worker.result(0)[0] and len(seen) == 2
    finally:
        source._close_feedback()


@pytest.mark.parametrize('buffered', [False, True])
def test_clean_capture_never_warns(monkeypatch: pytest.MonkeyPatch, buffered: bool) -> None:
    release_environment(monkeypatch, buffered, True)
    camera = Camera(np.full(SIZE, 32, dtype=np.uint8), lifecycle=False)
    statuses: list[str] = []
    source = DirectShowSource(DeviceConfig('OBS Virtual Camera', 0, color_correction='601to709'),
        DETECTION_LIMIT_SEC, lambda: None, lambda *args: camera, verifier=lambda image: True,
        on_status=statuses.append)
    assert list(source)
    assert 'overlay_feedback' not in statuses


def test_capture_to_attention(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """公開DTOまでを生成し、本物の設定画面で札と案内文を確認する。"""
    from playwright.sync_api import sync_playwright, expect
    from src.phase_j.distribution_runtime import DistributionRuntime
    from src.phase_j.distribution_http import make_server
    import threading
    release_environment(monkeypatch, True, True)
    runtime = DistributionRuntime(tmp_path)
    camera = Camera(encoded_camera(marked_image()), lifecycle=False)
    def publish(phase: str) -> None:
        payload = dict(display={})
        input_hold(payload, dict(input_calibration=dict(phase=phase, progress=0), t_sec=0), 0, None)
        runtime.accept_snapshot(payload)
    source = DirectShowSource(DeviceConfig('OBS Virtual Camera', 0, color_correction='601to709'),
        STATIC_END, lambda: None, lambda *args: camera, verifier=lambda image: True, on_status=publish)
    list(source)
    assert runtime.diagnostics()['calibration'] == 'overlay_feedback'
    server = make_server(runtime, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as runner:
            import os
            browser = runner.chromium.launch(channel='msedge' if os.name == 'nt' else 'chromium', headless=True)
            page = browser.new_page()
            page.goto(f'http://127.0.0.1:{server.server_port}/settings')
            page.wait_for_function('window.distributionReady')
            expect(page.locator('#runtime-state')).to_have_text('要対応')
            expect(page.locator('#camera-step-status')).to_contain_text('表示が入力の映像に映り込んでいます')
            expect(page.locator('#input-reason')).to_contain_text('ソース')
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
