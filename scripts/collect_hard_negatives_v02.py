"""
video_02 の 50 試合から hard negative 候補を集める。

各試合の中間時点 (start + duration/2) のフレームをサンプリングし、
そのフレームの全 board セルから「赤 false negative 候補」を抽出する。

具体的処理:
    1. matches.tsv から 50 試合の (start, end) を読む
    2. 各試合の中間秒で動画から 1 フレーム抽出
    3. 全 cell に対して CNN 分類 + HSV 赤度を計算
    4. CNN 予測≠赤 かつ HSV 赤度高 → 偽陰性候補
    5. すべての候補をグリッド画像に集約 + 個別保存

使い方:
    ./venv/bin/python scripts/collect_hard_negatives_v02.py
    出力: data/verify/hard_neg_v02/grid.png + individual/*.png + report.tsv
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
    BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY,
    COLOR_PURPLE, COLOR_RED, COLOR_YELLOW, HIDDEN_ROWS,
)
from src.calibration import CalibratedConfig
from src.match_state import MatchStateDetector
from src.patch_classifier import CnnPatchClassifier, GatedCnnClassifier
from scripts.extract_red_fn import _red_ratio, TARGET_CLASSES

VIDEO = Path("data/frames/video_02.mp4")
MATCHES_TSV = Path("data/verify/match_boundaries_v4/video_02/matches.tsv")
OUT_DIR = Path("data/verify/hard_neg_v02")
INDIV_DIR = OUT_DIR / "individual"

PATCH_DISPLAY = 96
LABEL_HEIGHT = 26
GRID_COLS = 8
RED_RATIO_THRESHOLD = 0.45  # この比率以上で赤偽陰性候補

CLASS_NAME = {
    COLOR_BLUE: "青", COLOR_PURPLE: "紫", COLOR_YELLOW: "黄",
    3: "緑", COLOR_RED: "赤", 0: "空",
}


def _make_tile(patch: np.ndarray, caption: str) -> np.ndarray:
    tile = np.full((PATCH_DISPLAY + LABEL_HEIGHT, PATCH_DISPLAY, 3), 32, dtype=np.uint8)
    resized = cv2.resize(patch, (PATCH_DISPLAY, PATCH_DISPLAY), interpolation=cv2.INTER_NEAREST)
    tile[:PATCH_DISPLAY, :, :] = resized
    for i, ln in enumerate(caption.split("\n")[:2]):
        cv2.putText(tile, ln,
                    (4, PATCH_DISPLAY + 12 + i * 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (240, 240, 240), 1, cv2.LINE_AA)
    return tile


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INDIV_DIR.mkdir(parents=True, exist_ok=True)

    if not MATCHES_TSV.exists():
        print(f"matches.tsv なし: {MATCHES_TSV}", file=sys.stderr)
        return 1

    cnn = CnnPatchClassifier.load(Path("models/cnn_global_best.pt"))
    config = CalibratedConfig.load(Path("models/calibration_video01.json"))
    gated = GatedCnnClassifier(color_classifier=cnn)
    match_det = MatchStateDetector(config.p1_region, config.p2_region)

    cap = cv2.VideoCapture(str(VIDEO))

    # matches.tsv 読み込み
    lines = MATCHES_TSV.read_text(encoding="utf-8").splitlines()[1:]
    matches: list[tuple[int, float, float]] = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        idx = int(parts[0])
        start = float(parts[1])
        end = float(parts[2])
        matches.append((idx, start, end))
    print(f"試合数: {len(matches)}")

    all_hits: list[tuple[int, str, int, int, int, float, np.ndarray]] = []
    # (match_idx, side, row, col, cnn_pred, red_ratio, patch)

    for match_idx, start, end in matches:
        t = (start + end) / 2.0  # 中間時刻
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)

        # 試合中フレームでなければスキップ
        if not match_det.is_in_match(frame):
            continue

        for side, region in (("1P", config.p1_region), ("2P", config.p2_region)):
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
                    if ratio < RED_RATIO_THRESHOLD:
                        continue
                    all_hits.append((match_idx, side, row, col, pred, ratio, patch))
        print(f"  match #{match_idx}: t={t:.0f}s 候補累計={len(all_hits)}")

    cap.release()

    print(f"\n総候補数: {len(all_hits)}")
    if not all_hits:
        print("候補なし")
        return 0

    # ratio 降順で上位 80 件
    all_hits.sort(key=lambda x: x[5], reverse=True)
    selected = all_hits[:80]

    # 個別保存
    for i, (m, side, row, col, pred, ratio, patch) in enumerate(selected):
        fname = f"{i:03d}_m{m:02d}_{side}_r{row:02d}_c{col}_pred{CLASS_NAME.get(pred,'?')}_r{ratio:.2f}.png"
        enlarged = cv2.resize(patch, (patch.shape[1] * 4, patch.shape[0] * 4),
                               interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(INDIV_DIR / fname), enlarged)

    # グリッド
    tiles = [
        _make_tile(p, f"m{m:02d}_{side}_r{row:02d}c{col}\npred={CLASS_NAME.get(pred,'?')} R={ratio:.2f}")
        for m, side, row, col, pred, ratio, p in selected
    ]
    tile_h = PATCH_DISPLAY + LABEL_HEIGHT
    rows_n = (len(tiles) + GRID_COLS - 1) // GRID_COLS
    grid = np.full((rows_n * tile_h, GRID_COLS * PATCH_DISPLAY, 3), 16, dtype=np.uint8)
    for idx, t_img in enumerate(tiles):
        r = idx // GRID_COLS
        c = idx % GRID_COLS
        grid[r * tile_h:(r + 1) * tile_h, c * PATCH_DISPLAY:(c + 1) * PATCH_DISPLAY] = t_img
    grid_path = OUT_DIR / "grid.png"
    cv2.imwrite(str(grid_path), grid)
    print(f"グリッド: {grid_path}")

    # TSV
    with (OUT_DIR / "report.tsv").open("w", encoding="utf-8") as f:
        f.write("idx\tmatch\tside\trow\tcol\tcnn_pred\tred_ratio\n")
        for i, (m, side, row, col, pred, ratio, _) in enumerate(selected):
            f.write(f"{i}\t{m}\t{side}\t{row}\t{col}\t{CLASS_NAME.get(pred,'?')}\t{ratio:.3f}\n")
    print(f"レポート: {OUT_DIR / 'report.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
