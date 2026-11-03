"""合成時系列で数フレームを描いて確認する (画像は local_frames のみ。コミットしない)。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.b3_review_panel import GraphSeries, draw_b3_panel
from src.b3_review_synth import synth_rows

OUT_DIR = Path("/mnt/d/puyo_analyzer/local_frames/b3_review_test")
SAMPLE_TIMES = (1.0, 3.5, 10.0, 18.0)
BASE_SIZE = (1280, 720)


def main() -> None:
    """黒地フレームに帯を連結し、4 時点の PNG を出す。"""
    rows = synth_rows()
    series = GraphSeries.from_rows(rows)
    times = [r["t_sec"] for r in rows]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for t in SAMPLE_TIMES:
        row = [r for r in rows if r["t_sec"] <= t][-1]
        frame = np.full((BASE_SIZE[1], BASE_SIZE[0], 3), 40, np.uint8)
        out = draw_b3_panel(frame, row, series, t)
        cv2.imwrite(str(OUT_DIR / f"synth_t{t:05.1f}.png"), out)
        print(t, out.shape, len(times))


if __name__ == "__main__":
    main()
