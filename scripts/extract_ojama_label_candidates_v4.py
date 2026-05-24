"""ラベリング用 grid v4 — v1+v2+v3 既使用を除外して新規 12 枚を抽出。"""
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

from scripts.extract_ojama_label_candidates_v2 import (
    LARGE_CHAIN_DELTA_MIN,
    N_CANDIDATES,
    SAMPLE_OFFSET_SEC,
    build_frame_panel,
    extract_cells,
    find_large_chain_events,
    get_frame,
)

CACHE_PATH = Path("data/training/score_series_cache.json")
EXISTING_INDEX_PATHS = [
    Path("data/verify/ojama_label_index.tsv"),
    Path("data/verify/ojama_label_index_v2.tsv"),
    Path("data/verify/ojama_label_index_v3.tsv"),
]
OUT_GRID = Path("data/verify/ojama_label_grid_v4.png")
OUT_INDEX = Path("data/verify/ojama_label_index_v4.tsv")
VIDEO_DIR = Path("data/frames")
FRAME_GAP = 12


def load_existing_match_keys() -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for p in EXISTING_INDEX_PATHS:
        if not p.is_file():
            continue
        with open(p) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                keys.add((r["video"], r["match"]))
    return keys


def main() -> int:
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    existing = load_existing_match_keys()
    print(f"既使用試合 (v1+v2+v3): {len(existing)}")

    events = find_large_chain_events(cache)
    filtered = [ev for ev in events if (ev[0], ev[1]) not in existing]
    print(f"大連鎖イベント (除外後): {len(filtered)}")

    selected: list = []
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
        for ci in range(6):
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
            fieldnames=["frame_idx", "t_sec", "video", "match", "side", "cell_idx"],
        )
        w.writeheader()
        w.writerows(index_rows)
    print(f"\n出力: {OUT_GRID} (shape={grid.shape})")
    print(f"index: {OUT_INDEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
