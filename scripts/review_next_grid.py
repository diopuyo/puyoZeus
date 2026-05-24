"""
video_02 の複数試合からネクスト (NEXT) のみを抽出し、グリッド表示する。
"""
from __future__ import annotations
import os, sys, csv
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

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

VIDEO = Path("data/frames/video_02.mp4")
TSV = Path("data/verify/match_boundaries_v4/video_02/matches.tsv")
OUT = Path("data/verify/dnext_review")

TILE_SZ = 180
LABEL_H = 40
GAP = 10


def annotate(patch: np.ndarray, label: str, color_code: int) -> np.ndarray:
    name, bgr = NAME_BGR.get(color_code, (f"?{color_code}", (128, 128, 128)))
    canvas = np.full((TILE_SZ + LABEL_H, TILE_SZ, 3), 32, dtype=np.uint8)
    resized = cv2.resize(patch, (TILE_SZ, TILE_SZ), interpolation=cv2.INTER_NEAREST)
    canvas[:TILE_SZ, :] = resized
    cv2.rectangle(canvas, (0, TILE_SZ), (TILE_SZ, TILE_SZ + LABEL_H), bgr, -1)
    text_color = (0, 0, 0) if sum(bgr) > 380 else (255, 255, 255)
    cv2.putText(canvas, f"{label}={name}", (4, TILE_SZ + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with TSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)

    det = NextDetector.load_default()
    cap = cv2.VideoCapture(str(VIDEO))

    if len(rows) >= 12:
        step = len(rows) // 12
        sampled = [rows[i * step] for i in range(12)]
    else:
        sampled = rows

    def _read_at(t_sec: float):
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
        ok, f = cap.read()
        if not ok or f is None:
            return None
        if f.shape[:2] != (1080, 1920):
            f = cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_AREA)
        return f

    pairs = []
    for r in sampled:
        idx = int(r["idx"])
        start, end = float(r["start_sec"]), float(r["end_sec"])
        t_mid = (start + end) / 2.0
        chosen_t: float | None = None
        chosen_result = None
        chosen_patches = None
        for offset in [0.0, 0.5, 1.0, -0.5, 1.5, 2.0, -1.0]:
            t = t_mid + offset
            if t < start or t > end:
                continue
            frames3 = []
            for d in [0.0, 0.5, 1.0]:
                f = _read_at(t + d)
                if f is None:
                    break
                frames3.append(f)
            if len(frames3) < 3:
                continue
            stable = det.detect_stable(frames3)
            if stable is not None:
                chosen_t = t
                chosen_result = stable
                chosen_patches = det.extract_patches(frames3[0])
                break

        if chosen_t is None or chosen_result is None or chosen_patches is None:
            pairs.append({
                "match": idx, "t": int(t_mid), "stable": False,
                "next_top": None, "next_bot": None,
            })
            continue

        pairs.append({
            "match": idx,
            "t": int(chosen_t),
            "stable": True,
            "next_top": (chosen_patches["next_top"], chosen_result.next_top),
            "next_bot": (chosen_patches["next_bot"], chosen_result.next_bot),
        })
    cap.release()

    n_rows = len(pairs)
    label_w = 160
    grid_w = label_w + 2 * TILE_SZ + GAP * 3
    row_h = TILE_SZ + LABEL_H + GAP
    grid_h = row_h * n_rows

    grid = np.full((grid_h, grid_w, 3), 16, dtype=np.uint8)

    for i, item in enumerate(pairs):
        y0 = i * row_h
        bg_color = (40, 40, 80) if item.get("stable") else (60, 30, 30)
        cv2.rectangle(grid, (0, y0), (label_w, y0 + TILE_SZ + LABEL_H), bg_color, -1)
        cv2.putText(grid, f"match {item['match']:>2}",
                    (8, y0 + TILE_SZ // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(grid, f"t={item['t']:>4}s",
                    (8, y0 + TILE_SZ // 2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        if not item.get("stable"):
            cv2.putText(grid, "UNSTABLE",
                        (8, y0 + TILE_SZ // 2 + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1, cv2.LINE_AA)
            continue
        x0 = label_w + GAP
        grid[y0:y0 + TILE_SZ + LABEL_H, x0:x0 + TILE_SZ] = annotate(
            item["next_top"][0], "NEXT_TOP", item["next_top"][1])
        x0 = label_w + GAP * 2 + TILE_SZ
        grid[y0:y0 + TILE_SZ + LABEL_H, x0:x0 + TILE_SZ] = annotate(
            item["next_bot"][0], "NEXT_BOT", item["next_bot"][1])

    out_path = OUT / "next_grid.png"
    cv2.imwrite(str(out_path), grid)
    print(f"出力: {out_path}")
    print(f"  {n_rows} 試合、各 2 セル (NEXT)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
