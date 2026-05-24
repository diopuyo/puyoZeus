"""
赤の偽陰性候補を抽出する診断ツール。

「赤判定はされていないが、中央画素の HSV 上で赤色相が支配的な」パッチを
スコア順に並べて可視化する。CNN が赤を青/紫/黄に誤認しているケースを捕まえる。

使い方:
    ./venv/bin/python scripts/extract_red_fn.py
    ./venv/bin/python scripts/extract_red_fn.py --top 40

出力:
    data/verify/red_fn/grid.png
    data/verify/red_fn/individual/*.png
    data/verify/red_fn/report.txt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
    HIDDEN_ROWS,
)
from src.calibration import CalibratedConfig
from src.patch_classifier import CnnPatchClassifier, GatedCnnClassifier

# 画素が「赤寄り」と見なす HSV 条件（H in 0-10 or 170-179, S>=80, V>=80）
RED_HUE_RANGES: tuple[tuple[int, int], ...] = ((0, 10), (170, 179))
RED_SAT_MIN: int = 80
RED_VAL_MIN: int = 80

# スコア計算対象のクラス（これらのなかに赤見逃しがあり得る）
TARGET_CLASSES: frozenset[int] = frozenset({
    COLOR_BLUE, COLOR_PURPLE, COLOR_YELLOW,
})

# 出力サイズ
PATCH_DISPLAY: int = 96
LABEL_HEIGHT: int = 26
GRID_COLS: int = 8

OUT_DIR: Path = Path("data/verify/red_fn")
INDIV_DIR: Path = OUT_DIR / "individual"

DEFAULT_CNN: Path = Path("models/cnn_global_best.pt")
DEFAULT_CALIB: Path = Path("models/calibration_video01.json")
CLASS_NAME: dict[int, str] = {
    COLOR_EMPTY: "空", COLOR_RED: "赤", COLOR_BLUE: "青",
    COLOR_YELLOW: "黄", COLOR_PURPLE: "紫", COLOR_OJAMA: "お", 3: "緑",
}


def _red_ratio(bgr_patch: np.ndarray) -> float:
    """中央 70% 領域で赤 hue を持つ画素の割合。"""
    h, w = bgr_patch.shape[:2]
    mh, mw = int(h * 0.15), int(w * 0.15)
    center = bgr_patch[mh:h-mh, mw:w-mw]
    if center.size == 0:
        return 0.0
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    mask = np.zeros_like(h_ch, dtype=bool)
    for h_lo, h_hi in RED_HUE_RANGES:
        mask |= (h_ch >= h_lo) & (h_ch <= h_hi)
    mask &= (s_ch >= RED_SAT_MIN) & (v_ch >= RED_VAL_MIN)
    total = h_ch.size
    return float(mask.sum()) / total


def _scan_frame_for_fn(
    frame_path: Path,
    gated: GatedCnnClassifier,
    config: CalibratedConfig,
) -> list[tuple[str, str, int, int, int, float, np.ndarray]]:
    """
    1 フレームの両サイドで、CNN 予測が非赤かつ赤度が高いパッチを返す。

    Returns: [(frame_stem, side, row, col, pred, red_ratio, patch), ...]
    """
    frame = cv2.imread(str(frame_path))
    if frame is None or frame.shape[:2] != (1080, 1920):
        return []
    hits: list[tuple[str, str, int, int, int, float, np.ndarray]] = []
    for side_name, region in (("1P", config.p1_region), ("2P", config.p2_region)):
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2c <= x1c or y2c <= y1c:
                    continue
                patch = frame[y1c:y2c, x1c:x2c].copy()
                pred = gated.classify(patch)
                if pred not in TARGET_CLASSES:
                    continue
                ratio = _red_ratio(patch)
                if ratio < 0.15:
                    continue
                hits.append((frame_path.stem, side_name, row, col, pred, ratio, patch))
    return hits


def _make_tile(patch: np.ndarray, caption: str) -> np.ndarray:
    tile = np.full((PATCH_DISPLAY + LABEL_HEIGHT, PATCH_DISPLAY, 3), 32, dtype=np.uint8)
    resized = cv2.resize(patch, (PATCH_DISPLAY, PATCH_DISPLAY), interpolation=cv2.INTER_NEAREST)
    tile[:PATCH_DISPLAY, :, :] = resized
    # 2 行キャプション
    lines = caption.split("\n")
    for i, ln in enumerate(lines[:2]):
        cv2.putText(
            tile, ln,
            (4, PATCH_DISPLAY + 12 + i * 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return tile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=40, help="上位何件保存するか")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INDIV_DIR.mkdir(parents=True, exist_ok=True)

    cnn = CnnPatchClassifier.load(DEFAULT_CNN)
    config = CalibratedConfig.load(DEFAULT_CALIB)
    gated = GatedCnnClassifier(color_classifier=cnn)

    all_hits: list[tuple[str, str, int, int, int, float, np.ndarray]] = []
    sample_dir = Path("data/frames/sample")
    frames = sorted(p for p in sample_dir.glob("frame_*.png") if "debug" not in p.name)
    for fp in frames:
        hits = _scan_frame_for_fn(fp, gated, config)
        all_hits.extend(hits)
        print(f"  {fp.name:24s}: 候補 {len(hits)} セル")

    all_hits.sort(key=lambda x: x[5], reverse=True)
    selected = all_hits[: args.top]
    print(f"\n全 {len(all_hits)} 件から赤度上位 {len(selected)} 件を保存")

    if not selected:
        print("候補なし")
        return 0

    for i, (stem, side, row, col, pred, ratio, patch) in enumerate(selected):
        pred_name = CLASS_NAME.get(pred, "?")
        fname = f"{i:03d}_{stem}_{side}_r{row:02d}_c{col}_pred{pred_name}_ratio{ratio:.2f}.png"
        enlarged = cv2.resize(
            patch,
            (patch.shape[1] * 4, patch.shape[0] * 4),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imwrite(str(INDIV_DIR / fname), enlarged)

    tiles = [
        _make_tile(
            patch,
            f"{stem[-5:]}_{side}_r{row:02d}c{col}\npred={CLASS_NAME.get(pred, '?')} R:{ratio:.2f}",
        )
        for stem, side, row, col, pred, ratio, patch in selected
    ]
    tile_h = PATCH_DISPLAY + LABEL_HEIGHT
    rows_n = (len(tiles) + GRID_COLS - 1) // GRID_COLS
    grid = np.full((rows_n * tile_h, GRID_COLS * PATCH_DISPLAY, 3), 16, dtype=np.uint8)
    for idx, t in enumerate(tiles):
        r = idx // GRID_COLS
        c = idx % GRID_COLS
        grid[r * tile_h:(r + 1) * tile_h, c * PATCH_DISPLAY:(c + 1) * PATCH_DISPLAY] = t
    grid_path = OUT_DIR / "grid.png"
    cv2.imwrite(str(grid_path), grid)
    print(f"グリッド: {grid_path}")

    report = OUT_DIR / "report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write(f"# 赤偽陰性候補 上位 {len(selected)} 件\n\n")
        for i, (stem, side, row, col, pred, ratio, _) in enumerate(selected):
            f.write(f"{i:03d}\t{stem}\t{side}\tr{row:02d}\tc{col}\tpred={CLASS_NAME.get(pred, '?')}\tred_ratio={ratio:.3f}\n")
    print(f"レポート: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
