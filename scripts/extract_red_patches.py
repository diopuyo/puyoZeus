"""
CNN が「赤」と判定した全パッチを抽出して可視化する診断ツール。

使い方:
    ./venv/bin/python scripts/extract_red_patches.py

出力:
    data/verify/red_patches/grid.png           ... 全「赤」判定パッチのグリッド表示
    data/verify/red_patches/individual/*.png   ... 個別パッチ（大きめ）
    data/verify/red_patches/report.txt         ... 位置メタデータ一覧

目的:
    CNN が何を「赤」と呼んでいるかを視覚的に確認し、赤の誤認/見逃しの
    原因（例: オレンジ気味の黄を赤扱い、暗所の赤を黄扱い等）を診断する。
"""
from __future__ import annotations

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
    COLOR_RED,
    HIDDEN_ROWS,
)
from src.calibration import CalibratedConfig
from src.image_reader import BoardRegion
from src.patch_classifier import CnnPatchClassifier, GatedCnnClassifier

# 出力ディレクトリ
OUT_DIR: Path = Path("data/verify/red_patches")
INDIV_DIR: Path = OUT_DIR / "individual"

# グリッド表示用: 各パッチを拡大
PATCH_DISPLAY: int = 96
LABEL_HEIGHT: int = 20
GRID_COLS: int = 8

# 優先モデル
_GLOBAL_BEST: Path = Path("models/cnn_global_best.pt")
_LATEST: Path = Path("models/cnn_best.pt")
DEFAULT_CNN: Path = _GLOBAL_BEST if _GLOBAL_BEST.exists() else _LATEST
DEFAULT_CALIB: Path = Path("models/calibration_video01.json")


def _scan_frame(
    frame_path: Path,
    gated: GatedCnnClassifier,
    region: BoardRegion,
    side_name: str,
) -> list[tuple[str, int, int, np.ndarray]]:
    """1 フレームの 1 側面を走査、赤と分類されたセルを返す。"""
    frame = cv2.imread(str(frame_path))
    if frame is None or frame.shape[:2] != (1080, 1920):
        return []
    hits: list[tuple[str, int, int, np.ndarray]] = []
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(frame.shape[1], x2)
            y2c = min(frame.shape[0], y2)
            if x2c <= x1c or y2c <= y1c:
                continue
            patch = frame[y1c:y2c, x1c:x2c].copy()
            if gated.classify(patch) == COLOR_RED:
                hits.append((side_name, row, col, patch))
    return hits


def _make_tile(
    patch: np.ndarray,
    caption: str,
) -> np.ndarray:
    """パッチ + キャプションのタイルを作る。"""
    tile_h = PATCH_DISPLAY + LABEL_HEIGHT
    tile = np.full((tile_h, PATCH_DISPLAY, 3), 32, dtype=np.uint8)
    resized = cv2.resize(patch, (PATCH_DISPLAY, PATCH_DISPLAY), interpolation=cv2.INTER_NEAREST)
    tile[:PATCH_DISPLAY, :, :] = resized
    cv2.putText(
        tile, caption,
        (4, PATCH_DISPLAY + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return tile


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INDIV_DIR.mkdir(parents=True, exist_ok=True)

    cnn_path = DEFAULT_CNN
    calib_path = DEFAULT_CALIB
    print(f"CNN  : {cnn_path}")
    print(f"Calib: {calib_path}")

    cnn = CnnPatchClassifier.load(cnn_path)
    config = CalibratedConfig.load(calib_path)
    gated = GatedCnnClassifier(color_classifier=cnn)

    sample_dir = Path("data/frames/sample")
    frames = sorted(
        p for p in sample_dir.glob("frame_*.png") if "debug" not in p.name
    )
    if not frames:
        print("フレームが見つかりません")
        return 1

    # 全赤判定パッチ収集
    all_hits: list[tuple[str, str, int, int, np.ndarray]] = []
    for fp in frames:
        for side, region in (("1P", config.p1_region), ("2P", config.p2_region)):
            hits = _scan_frame(fp, gated, region, side)
            for s, r, c, p in hits:
                all_hits.append((fp.stem, s, r, c, p))
            print(f"  {fp.name:24s} {side}: 赤判定 {len(hits)} セル")
    print(f"\n合計 {len(all_hits)} パッチ")

    if not all_hits:
        print("赤判定パッチがありません")
        return 0

    # 個別画像保存
    for i, (stem, side, row, col, patch) in enumerate(all_hits):
        out = INDIV_DIR / f"{i:03d}_{stem}_{side}_r{row:02d}_c{col}.png"
        # 見やすく 4 倍にリサイズ
        enlarged = cv2.resize(
            patch,
            (patch.shape[1] * 4, patch.shape[0] * 4),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imwrite(str(out), enlarged)

    # グリッド画像（一望表示）
    tiles = [
        _make_tile(p, f"{stem[-5:]}_{side}_r{row:02d}c{col}")
        for stem, side, row, col, p in all_hits
    ]
    tile_h = PATCH_DISPLAY + LABEL_HEIGHT
    rows = (len(tiles) + GRID_COLS - 1) // GRID_COLS
    grid_h = rows * tile_h
    grid_w = GRID_COLS * PATCH_DISPLAY
    grid = np.full((grid_h, grid_w, 3), 16, dtype=np.uint8)
    for idx, t in enumerate(tiles):
        r = idx // GRID_COLS
        c = idx % GRID_COLS
        y0 = r * tile_h
        x0 = c * PATCH_DISPLAY
        grid[y0:y0 + tile_h, x0:x0 + PATCH_DISPLAY] = t
    grid_path = OUT_DIR / "grid.png"
    cv2.imwrite(str(grid_path), grid)
    print(f"\nグリッド画像: {grid_path}")

    # メタデータ
    report_path = OUT_DIR / "report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"# 赤判定パッチ一覧 ({len(all_hits)} 件)\n")
        f.write(f"# CNN: {cnn_path}\n\n")
        for i, (stem, side, row, col, _) in enumerate(all_hits):
            f.write(f"{i:03d}\t{stem}\t{side}\tr{row:02d}\tc{col}\n")
    print(f"メタデータ: {report_path}")
    print(f"個別画像 : {INDIV_DIR}/ ({len(all_hits)} 枚)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
