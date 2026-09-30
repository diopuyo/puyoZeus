"""B20: ObservedDeathDetector の全解像度スコアと低解像度スコアを、同一フレーム列で並べて保存する。

目的は「低解像度で予備判定しても、全解像度検出と決定が同一か」を全フレームで検証する材料作り。
起動は logs/live_b20/scan_launch.sh 経由 (setsid -f)。src/ は変更しない。
出力 npz の各行: frame, [full_p1, full_p2, half_p1, half_p2, quarter_p1, quarter_p2]。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from src.exchange_event_terminal import TEMPLATE
from src.match_end_detector import SEARCH_P1, SEARCH_P2

NATIVE_SIZE = (1920, 1080)
STRIDE = 2  # 本番の認識間引き (60fps→30Hz)
SCALES = (0.5, 0.25)
FRAMES = '/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/frames/'


def score(roi: np.ndarray, template: np.ndarray) -> float:
    return float(cv2.minMaxLoc(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED))[1])


def shrink(image: np.ndarray, scale: float) -> np.ndarray:
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def scan(video: str, start: float, end: float, output: Path) -> None:
    cv2.setNumThreads(1)
    template = cv2.imread(str(TEMPLATE), cv2.IMREAD_GRAYSCALE)
    small = {s: shrink(template, s) for s in SCALES}
    cap = cv2.VideoCapture(FRAMES + video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    first = int(start * fps)
    last = min(int(end * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    index, rows, began = first, [], time.time()
    while index < last:
        ok, image = cap.read()
        if not ok:
            break
        for _ in range(STRIDE - 1):
            cap.grab()
        if (image.shape[1], image.shape[0]) != NATIVE_SIZE:
            image = cv2.resize(image, NATIVE_SIZE, interpolation=cv2.INTER_AREA)
        row = [float(index)] + [0.0] * (2 * (1 + len(SCALES)))
        for side, (x, y, w, h) in enumerate((SEARCH_P1, SEARCH_P2)):
            gray = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            row[1 + side] = score(gray, template)
            for k, scale in enumerate(SCALES):
                row[3 + 2 * k + side] = score(shrink(gray, scale), small[scale])
        rows.append(row)
        index += STRIDE
        if len(rows) % 5000 == 0:
            print(video, index, len(rows), 'elapsed %.0fs' % (time.time() - began), flush=True)
    np.savez_compressed(output, data=np.array(rows, dtype=np.float32), fps=fps, video=video)
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
