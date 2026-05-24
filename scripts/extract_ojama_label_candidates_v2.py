"""ラベリング用 grid v2 — 既存ラベル済み試合を除外して新規フレーム 12 枚を抽出する。

入力:
    data/training/score_series_cache.json
    data/verify/ojama_label_index.tsv  (既使用試合の除外用)

出力:
    data/verify/ojama_label_grid_v2.png
    data/verify/ojama_label_index_v2.tsv
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.ojama_warning import (
    CELL_COUNT,
    CELL_WIDTH,
    P1_BOARD_X,
    P2_BOARD_X,
    WARNING_BOTTOM_Y,
    WARNING_TOP_Y,
)

CACHE_PATH = Path("data/training/score_series_cache.json")
EXISTING_INDEX_PATH = Path("data/verify/ojama_label_index.tsv")
OUT_GRID = Path("data/verify/ojama_label_grid_v2.png")
OUT_INDEX = Path("data/verify/ojama_label_index_v2.tsv")
VIDEO_DIR = Path("data/frames")
EXPECTED_FRAME_SHAPE: tuple[int, int] = (1080, 1920)

LARGE_CHAIN_DELTA_MIN: int = 1000
N_CANDIDATES: int = 12
SAMPLE_OFFSET_SEC: float = 2.0
SCALE: int = 3
LABEL_BAR_H: int = 20
CELL_GAP: int = 4
FRAME_GAP: int = 12


def load_existing_match_keys() -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not EXISTING_INDEX_PATH.is_file():
        return keys
    with open(EXISTING_INDEX_PATH) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            keys.add((r["video"], r["match"]))
    return keys


def find_large_chain_events(cache: dict) -> list[tuple[str, str, float, str, int]]:
    events: list[tuple[str, str, float, str, int]] = []
    for vid, matches in cache.items():
        for midx, samples in matches.items():
            valid = [s for s in samples
                     if s["1p"] is not None and s["2p"] is not None]
            if len(valid) < 2:
                continue
            for i in range(1, len(valid)):
                d1 = valid[i]["1p"] - valid[i - 1]["1p"]
                d2 = valid[i]["2p"] - valid[i - 1]["2p"]
                if d1 >= LARGE_CHAIN_DELTA_MIN:
                    events.append((vid, midx, valid[i]["t"], "1P", d1))
                if d2 >= LARGE_CHAIN_DELTA_MIN:
                    events.append((vid, midx, valid[i]["t"], "2P", d2))
    events.sort(key=lambda x: x[4], reverse=True)
    return events


def get_frame(video_path: Path, t_sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != EXPECTED_FRAME_SHAPE:
        frame = cv2.resize(
            frame,
            (EXPECTED_FRAME_SHAPE[1], EXPECTED_FRAME_SHAPE[0]),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def extract_cells(frame: np.ndarray, side: str) -> list[np.ndarray]:
    base_x = P1_BOARD_X if side == "1P" else P2_BOARD_X
    out: list[np.ndarray] = []
    for i in range(CELL_COUNT):
        x1 = base_x + i * CELL_WIDTH
        x2 = x1 + CELL_WIDTH
        y1, y2 = WARNING_TOP_Y, WARNING_BOTTOM_Y
        out.append(frame[y1:y2, x1:x2].copy())
    return out


def annotate_cell(cell: np.ndarray, label: str, scale: int = SCALE) -> np.ndarray:
    h, w = cell.shape[:2]
    big = cv2.resize(cell, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    bar = np.zeros((LABEL_BAR_H, big.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, label, (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return np.vstack([bar, big])


def build_frame_panel(frame_idx: int, title: str,
                      cells_1p: list[np.ndarray],
                      cells_2p: list[np.ndarray]) -> np.ndarray:
    a1 = [annotate_cell(c, f"F{frame_idx} 1P S{i}")
          for i, c in enumerate(cells_1p)]
    a2 = [annotate_cell(c, f"F{frame_idx} 2P S{i}")
          for i, c in enumerate(cells_2p)]
    sep = np.full((a1[0].shape[0], CELL_GAP, 3), 60, dtype=np.uint8)
    big_sep = np.full((a1[0].shape[0], 12, 3), 120, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for c in a1:
        parts.append(c); parts.append(sep)
    parts.append(big_sep)
    for c in a2:
        parts.append(c); parts.append(sep)
    body = np.hstack(parts[:-1])
    title_bar = np.zeros((24, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, title, (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
    return np.vstack([title_bar, body])


def main() -> int:
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    existing = load_existing_match_keys()
    print(f"既使用試合: {len(existing)}")

    events = find_large_chain_events(cache)
    # 既使用試合を除外
    filtered = [ev for ev in events if (ev[0], ev[1]) not in existing]
    print(f"大連鎖イベント (除外後): {len(filtered)}")

    selected: list[tuple[str, str, float, str, int]] = []
    seen: set[tuple[str, str]] = set()
    for ev in filtered:
        key = (ev[0], ev[1])
        if key in seen:
            continue
        seen.add(key)
        selected.append(ev)
        if len(selected) >= N_CANDIDATES:
            break
    print(f"採用: {len(selected)}")

    panels: list[np.ndarray] = []
    index_rows: list[dict] = []
    for fi, (vid, midx, t_sec, side, delta) in enumerate(selected):
        sample_t = t_sec + SAMPLE_OFFSET_SEC
        frame = get_frame(VIDEO_DIR / f"{vid}.mp4", sample_t)
        if frame is None:
            print(f"  [skip] {vid} t={sample_t}")
            continue
        cells_1p = extract_cells(frame, "1P")
        cells_2p = extract_cells(frame, "2P")
        title = (f"F{fi}: {vid} match{midx} t={sample_t:.1f}s "
                 f"(chain by {side}, delta={delta})")
        panels.append(build_frame_panel(fi, title, cells_1p, cells_2p))
        for ci in range(CELL_COUNT):
            for s in ("1P", "2P"):
                index_rows.append({
                    "frame_idx": fi, "t_sec": round(sample_t, 2),
                    "video": vid, "match": midx,
                    "side": s, "cell_idx": ci,
                })
        print(f"  [ok] F{fi} {vid} m{midx} t={sample_t:.1f}s side={side} delta={delta}")

    sep = np.full((FRAME_GAP, panels[0].shape[1], 3), 30, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for p in panels:
        parts.append(p); parts.append(sep)
    grid = np.vstack(parts[:-1])
    cv2.imwrite(str(OUT_GRID), grid)

    with open(OUT_INDEX, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, delimiter="\t",
            fieldnames=["frame_idx", "t_sec", "video", "match",
                        "side", "cell_idx"],
        )
        w.writeheader()
        w.writerows(index_rows)

    print(f"\n出力:")
    print(f"  grid: {OUT_GRID} (shape={grid.shape})")
    print(f"  index: {OUT_INDEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
