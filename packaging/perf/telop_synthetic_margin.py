"""実データに可視テロップが無い (zenchi 211,008 frame で 0) ため、合成の陽性で「粗スコアと全解像度スコアの差」を測る。

実フレーム (PNG 連番) の背景へ、実テンプレートを劣化させて貼り (フェード・ぼかし・拡縮・明るさ・ノイズ・位置)、
全解像度の最大 NCC (本番 peak) と 1/4 縮小の粗い最大 NCC を並べる。陽性 (全解像度 >= 0.55-1e-4) のうち
粗スコアの最小値と、全解像度−粗 の最大乖離が、床 (PREFILTER_FLOOR) に対し十分な余白かを見る。
使い方 (cwd = 配布物の app/): python telop_synthetic_margin.py <PNG dir> <出力 json> [試行数]
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

from src.prepared_template import PreparedImage
from src.telop_detector import (COARSE_SCALE, DEFAULT_NCC_THRESHOLD, PREFILTER_FLOOR, SEARCH_H, SEARCH_W,
                                SEARCH_X, SEARCH_Y, TelopDetector, _shrink)

SEED = 20260930
NCC_MARGIN = 1e-4
ALPHAS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
BLURS = (0.0, 1.0, 2.0)
SCALES = (0.94, 0.97, 1.0, 1.03, 1.06)
BRIGHTNESS = (-40, -15, 0, 15, 40)
NOISES = (0.0, 4.0, 8.0)
MARGIN_PX = 10


def degrade(template: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scale = float(rng.choice(SCALES))
    image = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) if scale != 1.0 else template.copy()
    blur = float(rng.choice(BLURS))
    if blur:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    image = image.astype(np.float32)+float(rng.choice(BRIGHTNESS))
    noise = float(rng.choice(NOISES))
    if noise:
        image += rng.normal(0, noise, image.shape).astype(np.float32)
    return np.clip(image, 0, 255).astype(np.uint8)


def paste(background: np.ndarray, layer: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    height, width = layer.shape[:2]
    x = SEARCH_X+int(rng.integers(MARGIN_PX, SEARCH_W-width-MARGIN_PX))
    y = SEARCH_Y+int(rng.integers(MARGIN_PX, SEARCH_H-height-MARGIN_PX))
    frame = background.copy()
    patch = frame[y:y+height, x:x+width].astype(np.float32)
    rgb = layer if layer.ndim == 3 else np.repeat(layer[:, :, None], 3, axis=2)
    frame[y:y+height, x:x+width] = (patch*(1-alpha)+rgb.astype(np.float32)*alpha).astype(np.uint8)
    return frame


def measure(frame: np.ndarray, detector: TelopDetector) -> list[tuple[float, float]]:
    roi = cv2.cvtColor(frame[SEARCH_Y:SEARCH_Y+SEARCH_H, SEARCH_X:SEARCH_X+SEARCH_W], cv2.COLOR_BGR2GRAY)
    prepared, coarse_roi, rows = PreparedImage(roi), _shrink(roi), []
    for name in detector._templates:
        exact, _ = detector._prepared[name].peak(roi, DEFAULT_NCC_THRESHOLD, prepared)
        coarse = float(cv2.minMaxLoc(cv2.matchTemplate(coarse_roi, _shrink(detector._templates[name]), cv2.TM_CCOEFF_NORMED))[1])
        rows.append((float(exact), coarse))
    return rows


def main() -> None:
    frames_dir, output = Path(sys.argv[1]), Path(sys.argv[2])
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    rng = np.random.default_rng(SEED)
    cv2.setNumThreads(1)
    detector = TelopDetector.load_default(fast=False)
    files = sorted(frames_dir.glob('frame_*.png'))
    records = []
    for trial in range(trials):
        background = cv2.imread(str(files[int(rng.integers(0, len(files)))]))
        name = sorted(detector._templates)[int(rng.integers(0, len(detector._templates)))]
        layer = degrade(detector._templates[name], rng)
        frame = paste(background, layer, float(rng.choice(ALPHAS)), rng)
        for index, (exact, coarse) in enumerate(measure(frame, detector)):
            records.append((trial, index, exact, coarse))
    data = np.array(records)
    positive = data[:, 2] >= DEFAULT_NCC_THRESHOLD-NCC_MARGIN
    summary = dict(trials=trials, template_frame_pairs=int(len(data)), positive_pairs=int(positive.sum()),
                   floor=PREFILTER_FLOOR, coarse_scale=COARSE_SCALE,
                   min_coarse_on_positive=float(data[positive, 3].min()) if positive.any() else None,
                   false_skips_at_floor=int((data[positive, 3] < PREFILTER_FLOOR).sum()),
                   max_exact_minus_coarse_on_positive=float((data[positive, 2]-data[positive, 3]).max()) if positive.any() else None,
                   max_coarse_on_negative=float(data[~positive, 3].max()) if (~positive).any() else None,
                   coarse_percentiles_on_positive=[float(v) for v in np.percentile(data[positive, 3], [0, 1, 5, 50])] if positive.any() else None)
    output.write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
