"""複数動画ランダム時刻から 2P 側のネクスト・ダブルネクストをレビュー。"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.match_state import MatchStateDetector
from src.next_detector import NextDetector
from src.board import (
    COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_OJAMA,
    COLOR_PURPLE, COLOR_RED, COLOR_YELLOW,
)
from src.sampling_config import STABLE_FRAME_INTERVAL_SEC

NAME_BGR = {
    COLOR_EMPTY:  ("EMPTY",  (64, 64, 64)),
    COLOR_RED:    ("RED",    (0, 0, 220)),
    COLOR_BLUE:   ("BLUE",   (220, 80, 0)),
    COLOR_GREEN:  ("GREEN",  (0, 180, 0)),
    COLOR_YELLOW: ("YELLOW", (0, 200, 220)),
    COLOR_PURPLE: ("PURPLE", (180, 0, 180)),
    COLOR_OJAMA:  ("OJAMA",  (200, 200, 200)),
}

VIDEOS = [
    Path("data/frames/video_01.mp4"),
    Path("data/frames/video_02.mp4"),
    Path("data/frames/video_03.mp4"),
]
OUT_DIR = Path("data/verify/random_stable_review")

TILE_SZ = 150
LABEL_H = 36
GAP = 8
TARGET = 15
MAX_ATTEMPTS = 200
SEED = 42


def annotate(patch: np.ndarray, label: str, code: int) -> np.ndarray:
    name, bgr = NAME_BGR.get(code, (f"?{code}", (128, 128, 128)))
    canvas = np.full((TILE_SZ + LABEL_H, TILE_SZ, 3), 32, dtype=np.uint8)
    resized = cv2.resize(patch, (TILE_SZ, TILE_SZ), interpolation=cv2.INTER_NEAREST)
    canvas[:TILE_SZ, :] = resized
    cv2.rectangle(canvas, (0, TILE_SZ), (TILE_SZ, TILE_SZ + LABEL_H), bgr, -1)
    text_color = (0, 0, 0) if sum(bgr) > 380 else (255, 255, 255)
    cv2.putText(canvas, f"{label}={name}", (3, TILE_SZ + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
    return canvas


def _read_at(cap, t_sec: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, f = cap.read()
    if not ok or f is None:
        return None
    if f.shape[:2] != (1080, 1920):
        f = cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_AREA)
    return f


def _detect_2p_stable(det: NextDetector, frames: list[np.ndarray]):
    """2P 側で 2 連続フレーム判定一致なら返す。"""
    if not frames:
        return None
    results = [det.detect_2p(f) for f in frames]
    first = results[0]
    for r in results[1:]:
        if (r.next_top, r.next_bot, r.dnext_top, r.dnext_bot) != \
           (first.next_top, first.next_bot, first.dnext_top, first.dnext_bot):
            return None
    return first


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    det = NextDetector.load_default()
    match_det = MatchStateDetector.load_default()

    video_info: list[tuple[Path, float]] = []
    for v in VIDEOS:
        if not v.exists():
            print(f"skip: {v}")
            continue
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps
        cap.release()
        video_info.append((v, duration))
        print(f"{v.name}: {duration:.0f}s")

    if not video_info:
        return 1

    collected: list[dict] = []
    attempts = 0
    while len(collected) < TARGET and attempts < MAX_ATTEMPTS:
        attempts += 1
        v, duration = random.choice(video_info)
        t = random.uniform(duration * 0.1, duration * 0.9)
        cap = cv2.VideoCapture(str(v))
        frames_n = []
        for d in [0.0, STABLE_FRAME_INTERVAL_SEC]:
            f = _read_at(cap, t + d)
            if f is None:
                break
            frames_n.append(f)
        if len(frames_n) < 2:
            cap.release()
            continue
        if not match_det.is_in_match(frames_n[0]):
            cap.release()
            continue
        stable = _detect_2p_stable(det, frames_n)
        if stable is None:
            cap.release()
            continue
        patches = det.extract_patches(frames_n[0], side="2P")
        collected.append({
            "video": v.name, "t": int(t),
            "result": stable, "patches": patches,
        })
        cap.release()
        print(f"  [{len(collected)}/{TARGET}] {v.name} t={int(t)}s")

    print(f"\n採用 {len(collected)} / 試行 {attempts}")
    if not collected:
        return 1

    label_w = 180
    cells_per_row = 4
    cell_w = TILE_SZ
    grid_w = label_w + cells_per_row * cell_w + (cells_per_row + 1) * GAP
    row_h = TILE_SZ + LABEL_H + GAP
    grid_h = row_h * len(collected)
    grid = np.full((grid_h, grid_w, 3), 16, dtype=np.uint8)

    for i, item in enumerate(collected):
        y0 = i * row_h
        cv2.rectangle(grid, (0, y0), (label_w, y0 + TILE_SZ + LABEL_H),
                      (40, 40, 80), -1)
        cv2.putText(grid, f"#{i+1:>2}", (8, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(grid, item["video"][:18], (8, y0 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(grid, f"t={item['t']:>5}s", (8, y0 + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cells = [
            ("NEXT_TOP", item["patches"]["next_top"], item["result"].next_top),
            ("NEXT_BOT", item["patches"]["next_bot"], item["result"].next_bot),
            ("DNEXT_TOP", item["patches"]["dnext_top"], item["result"].dnext_top),
            ("DNEXT_BOT", item["patches"]["dnext_bot"], item["result"].dnext_bot),
        ]
        for j, (label, patch, code) in enumerate(cells):
            x0 = label_w + GAP + j * (cell_w + GAP)
            grid[y0:y0 + TILE_SZ + LABEL_H, x0:x0 + TILE_SZ] = annotate(patch, label, code)

    out_path = OUT_DIR / "random_stable_grid_2p.png"
    cv2.imwrite(str(out_path), grid)
    print(f"出力: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
