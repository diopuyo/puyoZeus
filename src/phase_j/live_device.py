"""DirectShow入力の明示選択と内容確認。色較正本体はここでは実行しない。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterator

import cv2
import numpy as np

from .live_source import CapturedFrame, FrameSource, NATIVE_SIZE, RECOGNITION_HZ
from .live_input_guard import FrameContinuity

ASPECT_TOLERANCE = 0.01
VERIFY_PERIOD_SEC = 1.0
READ_RETRY_SEC = 0.05
GRID_MARGIN = 20
GRID_POSITION_TOLERANCE = 80
BORDER_POSITION_TOLERANCE = 24
BORDER_COVERAGE = 0.65
CALIBRATION_ROOT = Path('config/device_calibration')


@dataclass(frozen=True)
class DeviceConfig:
    name: str
    index: int
    verification_only: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or type(self.index) is not int or self.index < 0:
            raise ValueError('機器名と非負の機器indexを明示してください')
        if type(self.verification_only) is not bool:
            raise ValueError('verification_onlyはboolが必要です')

    @classmethod
    def load(cls, path: Path) -> DeviceConfig:
        data = json.loads(path.read_text(encoding='utf-8'))
        return cls(name=data['name'], index=data['index'], verification_only=data.get('verification_only', True))

    @property
    def calibration_path(self) -> Path:
        identity = hashlib.sha256(f'{self.name}:{self.index}'.encode()).hexdigest()
        return CALIBRATION_ROOT / (identity + '.json')


class PuyoScreenVerifier:
    """固定の盤面枠位置と両側得点OCRを既存検出器で確認する。"""

    def __init__(self) -> None:
        from src.board_grid_detector import BoardGridDetector
        from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
        from src.score_ocr import ScoreOcr
        self.grid = BoardGridDetector()
        self.score = ScoreOcr.load_default()
        self.regions = (DEFAULT_P1_REGION, DEFAULT_P2_REGION)

    def __call__(self, image: np.ndarray) -> bool:
        for side, region in zip(('1P', '2P'), self.regions):
            x, y = region.x-GRID_MARGIN, region.y-GRID_MARGIN
            crop = image[y:region.y+region.height+GRID_MARGIN,
                         x:region.x+region.width+GRID_MARGIN]
            grid = self.grid.detect(crop)
            if self.score.read_side(image, side)[0] is None:
                return False
            if grid is None:
                if not frame_border_matches(crop, region.width, region.height):
                    return False
                continue
            expected = np.array([GRID_MARGIN, GRID_MARGIN,
                                 region.width+GRID_MARGIN, region.height+GRID_MARGIN])
            detected = np.array([*grid.top_left, *grid.bottom_right])
            if np.max(np.abs(expected-detected)) > GRID_POSITION_TOLERANCE:
                return False
        return True


def frame_border_matches(crop: np.ndarray, width: int, height: int) -> bool:
    """内部grid線が隠れる局面は、既存Hough分離器で外枠4辺を照合する。"""
    from src.board_grid_detector import (CANNY_LOW, CANNY_HIGH, HOUGH_RHO, HOUGH_THETA,
        HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP, _segments_to_lines)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    segments = cv2.HoughLinesP(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP)
    if segments is None:
        return False
    horizontal, vertical = _segments_to_lines(segments)
    return (border_pair(horizontal, 1, width, height) and border_pair(vertical, 0, height, width))


def border_pair(lines: list[tuple], axis: int, span: int, distance: int) -> bool:
    """位置と辺の長さの両方が合う線を、対向する2辺に要求する。"""
    return all(border_coverage(lines, axis, span, target) >= BORDER_COVERAGE
               for target in (GRID_MARGIN, GRID_MARGIN+distance))


def border_coverage(lines: list[tuple], axis: int, span: int, target: int) -> float:
    """ぷよや演出で分断された同じ辺の線分を重複なく合計する。"""
    intervals = []
    for line in lines:
        if abs((line[axis]+line[axis+2])/2-target) > BORDER_POSITION_TOLERANCE:
            continue
        start, end = sorted((line[1-axis], line[3-axis]))
        intervals.append((max(GRID_MARGIN, start), min(GRID_MARGIN+span, end)))
    covered, previous_end = 0.0, float('-inf')
    for start, end in sorted(intervals):
        covered += max(0.0, end-max(start, previous_end))
        previous_end = max(previous_end, end)
    return covered/span


class DirectShowSource(FrameSource):
    """名前だけでは検索しない。未確認映像は認識へ一枚も渡さない。"""

    def __init__(self, config: DeviceConfig, duration: float,
                 on_hold: Callable[[], None],
                 capture_factory: Callable[..., Any] = cv2.VideoCapture,
                 verifier: Callable[[np.ndarray], bool] | None = None,
                 clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep,
                 on_status: Callable[[str], None] | None = None) -> None:
        self.config, self.duration, self.on_hold = config, duration, on_hold
        self.capture_factory, self.verifier = capture_factory, verifier
        self.clock, self.sleep, self.dropped = clock, sleep, 0
        self.on_status = on_status

    def __iter__(self) -> Iterator[CapturedFrame]:
        capture = self.capture_factory(self.config.index, cv2.CAP_DSHOW)
        origin, last_verify, verified = self.clock(), float('-inf'), False
        continuity = FrameContinuity(VERIFY_PERIOD_SEC)
        try:
            verifier = self.verifier or PuyoScreenVerifier()
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, NATIVE_SIZE[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, NATIVE_SIZE[1])
            capture.set(cv2.CAP_PROP_FPS, RECOGNITION_HZ)
            while self.clock()-origin < self.duration:
                captured = self.clock()
                ok, image = capture.read() if capture.isOpened() else (False, None)
                normalized = self._normalize(image) if ok else None
                size = (image.shape[1], image.shape[0]) if normalized is not None else None
                transport_ready = normalized is not None and continuity.ready(normalized, size, captured)
                if not transport_ready:
                    verified = False
                    last_verify = float('-inf')
                elif captured-last_verify >= VERIFY_PERIOD_SEC:
                    verified, last_verify = verifier(normalized), captured
                if not verified:
                    if self.on_status:
                        self.on_status('verifying' if not transport_ready else 'no_puyo_screen')
                    else:
                        self.on_hold()
                    self.sleep(READ_RETRY_SEC)
                    continue
                elapsed = captured-origin
                yield CapturedFrame(round(elapsed*RECOGNITION_HZ), elapsed,
                                    captured, self.clock(), normalized, source_size=size)
                self.sleep(max(0.0, captured+1/RECOGNITION_HZ-self.clock()))
        finally:
            capture.release()

    @staticmethod
    def _normalize(image: np.ndarray | None) -> np.ndarray | None:
        if image is None or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            return None
        height, width = image.shape[:2]
        if abs(width/height-NATIVE_SIZE[0]/NATIVE_SIZE[1]) > ASPECT_TOLERANCE:
            return None
        return image if (width, height) == NATIVE_SIZE else cv2.resize(image, NATIVE_SIZE)


class DeviceMetadataCapture:
    """既存動画CLIへFPS/尺のみ渡す。実デバイスは認識processだけが開く。"""

    def __init__(self, duration: float) -> None:
        self.duration = duration

    def isOpened(self) -> bool:
        return True

    def get(self, key: int) -> float:
        return {cv2.CAP_PROP_FPS: RECOGNITION_HZ,
                cv2.CAP_PROP_FRAME_COUNT: self.duration*RECOGNITION_HZ,
                cv2.CAP_PROP_FRAME_WIDTH: NATIVE_SIZE[0],
                cv2.CAP_PROP_FRAME_HEIGHT: NATIVE_SIZE[1]}.get(key, 0.0)

    def set(self, key: int, value: float) -> bool:
        return True

    def release(self) -> None:
        pass
