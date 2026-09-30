"""B21: TelopDetector の全解像度スコア(本番の peak)と、縮小 ROI での粗スコアを同一フレーム列で並べて保存する。

目的: 「縮小した粗スコアが床未満なら、そのテンプレートは閾値 0.55 未満」と扱う予備判定が、全フレームで
決定 (is_visible / bbox) と同一かを、任意の床について後から計算できる材料を作る (fast_terminal の B20 と同方式)。
出力 npz `data` の各行: frame, [exact_t0, exact_t1, x_t0, y_t0, x_t1, y_t1, half_t0, half_t1, quarter_t0, quarter_t1]。
exact は本番 `PreparedTemplate.peak` の値 (閾値近傍では cv2.matchTemplate)。src/ は変更しない (読むだけ)。
起動は WSL の setsid -f + nice 19。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from src.prepared_template import PreparedImage
from src.telop_detector import SEARCH_H, SEARCH_W, SEARCH_X, SEARCH_Y, TelopDetector

NATIVE_SIZE = (1920, 1080)
STRIDE = 2  # 本番の認識間引き (60fps→30Hz)
SCALES = (0.5, 0.25)
FRAMES = '/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/frames/'
REPORT_EVERY = 5000
TEMPLATE_DIR = Path(FRAMES).parents[1] / 'models' / 'ui_templates'  # 作業ツリーに models が無いため本体側を絶対パスで読む


def shrink(image: np.ndarray, scale: float) -> np.ndarray:
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def coarse_peak(roi: np.ndarray, template: np.ndarray) -> float:
    return float(cv2.minMaxLoc(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED))[1])


def frame_row(index: int, image: np.ndarray, detector: TelopDetector, small: dict) -> list[float]:
    roi = image[SEARCH_Y:SEARCH_Y+SEARCH_H, SEARCH_X:SEARCH_X+SEARCH_W]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    prepared = PreparedImage(gray)
    row = [float(index)]
    exact, positions, coarse = [], [], {scale: [] for scale in SCALES}
    for name, template in detector._templates.items():
        peak, position = detector._prepared[name].peak(gray, detector._threshold, prepared)
        exact.append(peak)
        positions += list(position)
        for scale in SCALES:
            coarse[scale].append(coarse_peak(shrink(gray, scale), small[name][scale]))
    return row + exact + positions + coarse[0.5] + coarse[0.25]


def scan(video: str, start: float, end: float, output: Path) -> None:
    cv2.setNumThreads(1)
    detector = TelopDetector.load_default(TEMPLATE_DIR)
    small = {name: {scale: shrink(template, scale) for scale in SCALES} for name, template in detector._templates.items()}
    assert len(detector._templates) == 2, sorted(detector._templates)  # 列配置が 2 テンプレート前提
    cap = cv2.VideoCapture(FRAMES + video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    first = int(start*fps)
    last = min(int(end*fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    index, rows, began = first, [], time.time()
    while index < last:
        ok, image = cap.read()
        if not ok:
            break
        for _ in range(STRIDE-1):
            cap.grab()
        if (image.shape[1], image.shape[0]) != NATIVE_SIZE:
            image = cv2.resize(image, NATIVE_SIZE, interpolation=cv2.INTER_AREA)
        rows.append(frame_row(index, image, detector, small))
        index += STRIDE
        if len(rows) % REPORT_EVERY == 0:
            print(video, index, len(rows), 'elapsed %.0fs' % (time.time()-began), flush=True)
    np.savez_compressed(output, data=np.array(rows, dtype=np.float32), fps=fps, video=video,
                        names=np.array(sorted(detector._templates)))
    print('done', video, len(rows), 'frames', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--start', type=float, default=0.0)
    parser.add_argument('--end', type=float, default=1e9)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    scan(options.video, options.start, options.end, options.output)


if __name__ == '__main__':
    main()
