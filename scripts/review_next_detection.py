"""
複数フレームに対してネクスト検出を走らせ、結果を 1 枚のグリッド画像に集約。
人の目で「ROI 位置」と「色判定」の両方を確認できる。

使い方:
    ./venv/bin/python scripts/review_next_detection.py
    出力: data/verify/next_review/grid_<frame>.png  各フレーム別レビュー画像
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
    COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_OJAMA,
    COLOR_PURPLE, COLOR_RED, COLOR_YELLOW,
)
from src.next_detector import (
    NextDetector,
    ROI_1P_NEXT_TOP, ROI_1P_NEXT_BOT, ROI_1P_DNEXT_TOP, ROI_1P_DNEXT_BOT,
)

NAME_BGR: dict[int, tuple[str, tuple[int, int, int]]] = {
    COLOR_EMPTY:  ("empty",  (64, 64, 64)),
    COLOR_RED:    ("RED",    (0, 0, 220)),
    COLOR_BLUE:   ("BLUE",   (220, 80, 0)),
    COLOR_GREEN:  ("GREEN",  (0, 180, 0)),
    COLOR_YELLOW: ("YELLOW", (0, 200, 220)),
    COLOR_PURPLE: ("PURPLE", (180, 0, 180)),
    COLOR_OJAMA:  ("OJAMA",  (200, 200, 200)),
}


def annotate(patch: np.ndarray, label: str, color_code: int, target_size: int = 200) -> np.ndarray:
    """パッチを大きくしてラベルを下に貼る。"""
    name, bgr = NAME_BGR.get(color_code, (f"?{color_code}", (128, 128, 128)))
    h, w = patch.shape[:2]
    # 縦横とも target_size 以下に収まるようリサイズ
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    # 下にラベル帯
    label_h = 40
    canvas = np.full((target_size + label_h, target_size, 3), 32, dtype=np.uint8)
    # 中央配置
    yo = (target_size - new_h) // 2
    xo = (target_size - new_w) // 2
    canvas[yo:yo + new_h, xo:xo + new_w] = resized
    # ラベル
    cv2.rectangle(canvas, (0, target_size), (target_size, target_size + label_h), bgr, -1)
    text_color = (0, 0, 0) if sum(bgr) > 380 else (255, 255, 255)
    txt = f"{label}={name}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    tx = (target_size - tw) // 2
    cv2.putText(canvas, txt, (tx, target_size + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)
    return canvas


def make_review_image(frame: np.ndarray, det: NextDetector, frame_label: str) -> np.ndarray:
    result = det.detect(frame)
    patches = det.extract_patches(frame)
    cells = [
        ("NEXT_TOP", patches["next_top"], result.next_top),
        ("NEXT_BOT", patches["next_bot"], result.next_bot),
        ("DNEXT_TOP", patches["dnext_top"], result.dnext_top),
        ("DNEXT_BOT", patches["dnext_bot"], result.dnext_bot),
    ]
    tiles = [annotate(p, lbl, c) for lbl, p, c in cells]
    # 横一列、左に元 ROI 周辺の縦スリップを表示
    # 4 タイル × 200px = 800、左に元画像 ROI を 250px 表示
    roi_y = 100
    roi_strip = frame[roi_y:roi_y + 400, 690:830]
    roi_strip = cv2.resize(roi_strip, (250, 400), interpolation=cv2.INTER_AREA)
    # ヘッダ
    header_h = 40
    grid_h = 240 + header_h
    grid_w = 250 + 4 * 200 + 30  # left strip + 4 tiles + margin
    grid = np.full((grid_h, grid_w, 3), 16, dtype=np.uint8)
    cv2.putText(grid, f"{frame_label}: {result.next_pair} -> {result.dnext_pair}",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    # 上部 240 だけスリップ（フィット）
    strip_resized = cv2.resize(roi_strip, (200, 240), interpolation=cv2.INTER_AREA)
    grid[header_h:header_h + 240, 10:210] = strip_resized
    cv2.putText(grid, "ROI raw",
                (10, header_h + 240 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    for i, t in enumerate(tiles):
        x0 = 250 + i * 200
        grid[header_h:header_h + 240, x0:x0 + 200] = t
    return grid


def main() -> int:
    out_dir = Path("data/verify/next_review")
    out_dir.mkdir(parents=True, exist_ok=True)

    det = NextDetector.load_default()

    targets = [
        # video_01 (clean test set)
        "data/frames/sample/frame_0300s.png",
        "data/frames/sample/frame_0600s.png",
        "data/frames/sample/frame_0900s.png",
        "data/frames/sample/frame_1500s.png",
        "data/frames/sample/frame_2100s.png",
        "data/frames/sample/frame_2700s.png",
        "data/frames/sample/frame_3200s.png",
        # video_02 (different match)
        "data/frames/review_video_02/frame_0210s.png",
        "data/frames/review_video_02/frame_0240s.png",
        "data/frames/review_video_02/frame_0270s.png",
    ]
    for p in targets:
        path = Path(p)
        if not path.exists():
            print(f"skip: {p}")
            continue
        frame = cv2.imread(str(path))
        if frame is None or frame.shape[:2] != (1080, 1920):
            print(f"skip (resolution): {p}")
            continue
        img = make_review_image(frame, det, path.stem)
        out = out_dir / f"review_{path.stem}.png"
        cv2.imwrite(str(out), img)
        print(f"  {path.stem}: {out}")

    print(f"\n出力: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
