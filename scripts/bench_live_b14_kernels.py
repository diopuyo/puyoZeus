"""同じ実画像で各施策一回あたりのコストを交互計測する補助診断。"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np

from scripts.measure_realtime_breakdown_20260928 import DEFAULT_VIDEO
from src.background_fingerprint import CellFingerprint, CellPatchFingerprint, _compute_ncc_prepared
from src.image_reader import ImageReader, DEFAULT_P1_REGION, DEFAULT_P2_REGION, _median_hsv_3ch
from src.telop_detector import TelopDetector, SEARCH_X, SEARCH_Y, SEARCH_W, SEARCH_H
from src.prepared_template import PreparedImage
from src.ui_mask import UiMaskMatcher

REPEATS, SAMPLE_SEC, MILLISECONDS = 100, 2900, 1000.0
OUTPUT = Path('logs/live_b14/kernels.json')


def measure(before: Callable[[], Any], after: Callable[[], Any]) -> dict:
    timings: list[list[float]] = [[], []]
    for _ in range(REPEATS):
        for index, function in enumerate((before, after)):
            start = perf_counter()
            function()
            timings[index].append((perf_counter()-start)*MILLISECONDS)
    return dict(repeats=REPEATS, before_ms=np.percentile(timings[0], [50, 95]).tolist(),
                after_ms=np.percentile(timings[1], [50, 95]).tolist())


def previous_ncc(current: np.ndarray, centered: np.ndarray, dot: float) -> float:
    values = current.ravel().astype(np.float64)
    values.std()
    values -= values.mean()
    return float(np.dot(values, centered)/np.sqrt(np.dot(values, values)*dot))


def main() -> None:
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(DEFAULT_VIDEO))
    capture.set(cv2.CAP_PROP_POS_MSEC, SAMPLE_SEC*MILLISECONDS)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError('計測画像を取得できません')
    reader, ui = ImageReader.__new__(ImageReader), UiMaskMatcher.load_default()
    x1, y1, x2, y2 = DEFAULT_P1_REGION.cell_sample_rect(12, 2)
    patch = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    background = CellPatchFingerprint(hsv)
    centered = hsv.ravel().astype(np.float64)
    centered -= centered.mean()
    dot = float(np.dot(centered, centered))
    deviation = float(hsv.astype(np.float64).std())
    result = dict(sample_sec=SAMPLE_SEC, cpu_threads=1)
    result['background_v_median'] = measure(lambda: float(np.median(hsv[:, :, 2])),
                                             lambda: background.v_median)
    result['ncc_center_reuse'] = measure(lambda: previous_ncc(hsv, centered, dot),
                                       lambda: _compute_ncc_prepared(hsv, centered, dot, deviation))
    # EMPTY確定時に丸ごと省ける処理。空判定自身のコストはこの行に含めない。
    result['empty_median_distance'] = measure(
        lambda: CellFingerprint(*_median_hsv_3ch(hsv)).distance_to(CellFingerprint(0, 0, 0)), lambda: None)
    result['ui_inner_product'] = measure(lambda: ui.match(patch).is_ui, lambda: ui.is_ui(patch))
    result['board_hsv_roi'] = measure(lambda: cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                                     lambda: reader._board_hsv_frame(frame, (DEFAULT_P1_REGION, DEFAULT_P2_REGION)))
    detector = TelopDetector.load_default()
    roi = cv2.cvtColor(frame[SEARCH_Y:SEARCH_Y+SEARCH_H, SEARCH_X:SEARCH_X+SEARCH_W], cv2.COLOR_BGR2GRAY)
    result['telop_fft'] = measure(lambda: original_telop(detector, roi), lambda: prepared_telop(detector, roi))
    OUTPUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


def original_telop(detector: TelopDetector, roi: np.ndarray) -> None:
    for template in detector._templates.values():
        cv2.minMaxLoc(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED))


def prepared_telop(detector: TelopDetector, roi: np.ndarray) -> None:
    image = PreparedImage(roi)
    for template in detector._prepared.values():
        template.peak(roi, detector._threshold, image)


if __name__ == '__main__':
    main()
