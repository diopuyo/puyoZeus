"""ojama seed (olRyxDGacbg_ojama_seed_v3) から無作為50枚のモンタージュ画像を作る
(userレビュー用、使い捨てスクリプト)。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.self_supervised.label_store import LabelStore
from src.self_supervised.pseudo_label import COMPONENT_CELL

STORE_ROOT = _ROOT / "data/pseudo_labels_olRyxDGacbg_demo_2026-08-07"
VIDEO_ID = "olRyxDGacbg_ojama_seed_v3_strict2"
N_SAMPLES = 50
SEED = 42
CELL_PX = 80  # モンタージュ内 1 patch の描画サイズ
GRID_COLS = 10


def main() -> None:
    store = LabelStore(video_id=VIDEO_ID, root=STORE_ROOT)
    samples = list(store.load(COMPONENT_CELL))
    print(f"[montage] total samples = {len(samples)}")
    rng = random.Random(SEED)
    rng.shuffle(samples)
    picked = samples[:N_SAMPLES]

    n = len(picked)
    n_rows = (n + GRID_COLS - 1) // GRID_COLS
    canvas = np.full((n_rows * CELL_PX, GRID_COLS * CELL_PX, 3), 40, dtype=np.uint8)
    for i, s in enumerate(picked):
        patch = s.input_data["patch"]
        resized = cv2.resize(patch, (CELL_PX - 4, CELL_PX - 4), interpolation=cv2.INTER_NEAREST)
        r, c = divmod(i, GRID_COLS)
        y0, x0 = r * CELL_PX + 2, c * CELL_PX + 2
        canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized

    out_path = _ROOT / "data/verify/youtube_demo_2026-08-07/_ojama_seed_montage_strict2.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"[montage] saved {n} patches -> {out_path}")


if __name__ == "__main__":
    main()
