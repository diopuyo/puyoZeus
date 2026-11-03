"""候補の設置セル・上の落下セル・初段消去セルだけを画像で照合する。"""
from __future__ import annotations
from src.phase_j.live_color_roi import full_image
from src import prefire_work_budget as work_budget

from time import perf_counter
from typing import Any
import cv2
import numpy as np
from src.image_reader import ColorClassifier, DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.effect_glow_detector import BRIGHT_V_THRESHOLD

FRAME_SIZE = (1920, 1080)
MAX_FRAMES, MAX_CELLS, POP_SAMPLES = 6, 48, 4
COLS, HIDDEN_CELLS = 6, 6
WINDOW_SEC, BUDGET_SEC = .35, .5
TRACK_HEIGHTS = (1, 2)


def probe_cells(candidates: list[dict]) -> list[int]:
    groups = [[i for c in candidates for i in c['added_cells']],
              [i for c in candidates for i in c['pop_cells'][:POP_SAMPLES]]]
    groups.append([i - height*COLS for i in groups[0] for height in TRACK_HEIGHTS])
    return list(dict.fromkeys(i for group in groups for i in group if i >= HIDDEN_CELLS))[:MAX_CELLS]


def read_patches(frame: np.ndarray, side: int, cells: list[int], classifier: Any) -> dict:
    if (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
        frame = cv2.resize(full_image(frame), FRAME_SIZE, interpolation=cv2.INTER_AREA)
    region = (DEFAULT_P1_REGION, DEFAULT_P2_REGION)[side]
    values = {}
    for cell in cells:
        x1, y1, x2, y2 = region.cell_sample_rect(*divmod(cell, COLS))
        patch = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        values[cell] = (int(classifier.classify(patch)), float(np.mean(hsv[:, :, 2] >= BRIGHT_V_THRESHOLD)))
    return values


def summarize(candidate: dict, samples: list[tuple], trigger: float) -> dict:
    before = [(t, p) for t, p in samples if t < trigger]
    after = [(t, p) for t, p in samples if t >= trigger]
    added = candidate['added_cells']
    matches = [float(p[i][0] == candidate['origin'][i]) for _, p in before for i in added if i in p]
    # 同列で上から下へ近づいた一致だけを落下軌跡の支持とする。
    track = []
    for i in added:
        hits = [(t, i-j) for t, p in before for j in p if j % COLS == i % COLS and
                j <= i and p[j][0] == candidate['origin'][i] and i-j <= max(TRACK_HEIGHTS)*COLS]
        track.append(float(any(t1 < t2 and h1 > h2 for t1, h1 in hits for t2, h2 in hits)))
    glow = [max(0., p[i][1]-q[i][1]) for _, p in after for _, q in before
            for i in candidate['pop_cells'][:POP_SAMPLES] if i in p and i in q]
    return dict(hsv=float(np.mean(matches)) if matches else 0.,
                track=float(np.mean(track)) if track else 0., glow=max(glow, default=0.),
                image_available=bool(matches), image_frames=len(samples))


def image_evidence(candidates: list[dict], frames: list[tuple], side: int,
                   trigger: float, stamp: float, deadline: float | None = None) -> dict:
    """呼出側が保持した到着済み画像だけを使う。デコードはlive側の仕事。"""
    began = perf_counter()
    deadline = deadline if deadline is not None else began+BUDGET_SEC
    if any(t > stamp for t, _ in frames):
        raise ValueError('未到着の画像を起点推定へ渡せません')
    available = [(t, f) for t, f in frames if trigger-WINDOW_SEC <= t <= stamp]
    indices = np.linspace(0, len(available)-1, min(MAX_FRAMES, len(available)), dtype=int)
    cells, samples, classifier = probe_cells(candidates), [], ColorClassifier()
    for index in indices:
        if not work_budget.enabled() and perf_counter() >= deadline:
            break
        t, frame = available[index]
        samples.append((t, read_patches(frame, side, cells, classifier)))
    for candidate in candidates:
        candidate.update(summarize(candidate, samples, trigger))
    return dict(image_ms=(perf_counter()-began)*1000, cells=len(cells), frames=len(samples),
                latest_sec=max((t for t, _ in samples), default=None),
                budget_exhausted=len(samples) < len(indices))
