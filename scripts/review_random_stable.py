"""
複数動画のランダム時刻から 15 件の安定 3 フレーム判定データを集めて
ネクスト・ダブルネクストの 4 セルをグリッド表示する。

採用条件:
    - in-match フレーム（match_state HSV 判定）
    - 3 連続フレーム (0.5s 間隔) で next/dnext 4 セル全部の判定が一致
"""
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
TARGET_SAMPLES = 15
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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    det = NextDetector.load_default()
    match_det = MatchStateDetector.load_default()

    # 利用可能な動画の duration を取得
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
        print("No videos available", file=sys.stderr)
        return 1

    # サンプル収集
    collected: list[dict] = []
    attempts = 0
    while len(collected) < TARGET_SAMPLES and attempts < MAX_ATTEMPTS:
        attempts += 1
        v, duration = random.choice(video_info)
        # ランダム時刻（最初・最後を避けるため ±10% マージン）
        t = random.uniform(duration * 0.1, duration * 0.9)

        cap = cv2.VideoCapture(str(v))

        # 0.2s 間隔の 2 連続フレーム
        frames_n = []
        for d in [0.0, 0.2]:
            f = _read_at(cap, t + d)
            if f is None:
                break
            frames_n.append(f)

        if len(frames_n) < 2:
            cap.release()
            continue

        # 試合中フレーム判定
        if not match_det.is_in_match(frames_n[0]):
            cap.release()
            continue

        # 2 連続フレーム安定判定
        stable = det.detect_stable(frames_n)
        if stable is None:
            cap.release()
            continue

        # 採用
        patches = det.extract_patches(frames_n[0])
        collected.append({
            "video": v.name,
            "t": int(t),
            "result": stable,
            "patches": patches,
        })
        cap.release()
        print(f"  [{len(collected)}/{TARGET_SAMPLES}] {v.name} t={int(t)}s")

    print(f"\n採用 {len(collected)} / 試行 {attempts}")
    if not collected:
        print("採用ゼロ", file=sys.stderr)
        return 1

    # グリッド生成
    label_w = 180
    cells_per_row = 4  # NEXT_TOP, NEXT_BOT, DNEXT_TOP, DNEXT_BOT
    cell_w = TILE_SZ
    grid_w = label_w + cells_per_row * cell_w + (cells_per_row + 1) * GAP
    row_h = TILE_SZ + LABEL_H + GAP
    grid_h = row_h * len(collected)

    grid = np.full((grid_h, grid_w, 3), 16, dtype=np.uint8)

    for i, item in enumerate(collected):
        y0 = i * row_h
        # 左ラベル
        cv2.rectangle(grid, (0, y0), (label_w, y0 + TILE_SZ + LABEL_H),
                      (40, 40, 80), -1)
        cv2.putText(grid, f"#{i+1:>2}", (8, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(grid, item["video"][:18], (8, y0 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(grid, f"t={item['t']:>5}s", (8, y0 + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        # 4 セル
        cells = [
            ("NEXT_TOP", item["patches"]["next_top"], item["result"].next_top),
            ("NEXT_BOT", item["patches"]["next_bot"], item["result"].next_bot),
            ("DNEXT_TOP", item["patches"]["dnext_top"], item["result"].dnext_top),
            ("DNEXT_BOT", item["patches"]["dnext_bot"], item["result"].dnext_bot),
        ]
        for j, (label, patch, code) in enumerate(cells):
            x0 = label_w + GAP + j * (cell_w + GAP)
            tile = annotate(patch, label, code)
            grid[y0:y0 + TILE_SZ + LABEL_H, x0:x0 + TILE_SZ] = tile

    out_path = OUT_DIR / "random_stable_grid.png"
    cv2.imwrite(str(out_path), grid)
    print(f"出力: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
