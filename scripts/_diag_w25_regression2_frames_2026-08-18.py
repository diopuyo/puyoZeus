"""W25第3弾 新規悪化2シート機構特定: 実画面証拠フレーム抽出 (計装専用、src/無改修)。

対象セルに赤枠を描画した 2P 盤面クロップを保存する。
"""
from __future__ import annotations

from pathlib import Path

import cv2

from src.image_reader import DEFAULT_P2_REGION

VIDEO_DIR = Path.home() / "frames"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "verify" / "diag_w25_regression2_2026-08-18" / "frames"

ROW_PX = DEFAULT_P2_REGION.height / 12  # 可視12行
COL_PX = DEFAULT_P2_REGION.width / 6

TARGETS = [
    {
        "name": "c10",
        "video": VIDEO_DIR / "video_c10.mp4",
        "cells": [(8, 1), (10, 2)],
        "frames": [
            (15391, "before_cnn_switch"),
            (15401, "cnn_reports_9_tsumofall"),
            (15403, "wrong_landing_merge_conf1"),
            (15451, "off_selfrecovers_to9_ref"),
            (15517, "gt_frame"),
        ],
    },
    {
        "name": "c109",
        "video": VIDEO_DIR / "video_c109.mp4",
        "cells": [(3, 2)],
        "frames": [
            (652090, "color5_placed_normally"),
            (652452, "chain_cnn_reports_9"),
            (652480, "ojamafall_to_stable_cnn9"),
            (652500, "off_correct_merge_to9_ref"),
            (652546, "gt_frame"),
        ],
    },
]


def draw_cell_box(img, r: int, c: int):
    """可視行 r(1-12)・列 c(0-5) のセルに赤枠を描く (region 座標系)。"""
    x1 = int(DEFAULT_P2_REGION.x + c * COL_PX)
    x2 = int(DEFAULT_P2_REGION.x + (c + 1) * COL_PX)
    y1 = int(DEFAULT_P2_REGION.y + (r - 1) * ROW_PX)
    y2 = int(DEFAULT_P2_REGION.y + r * ROW_PX)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        cap = cv2.VideoCapture(str(target["video"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        for frame_idx, label in target["frames"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
            ok, frame = cap.read()
            if not ok:
                print(f"skip (read fail): {target['name']} {frame_idx}")
                continue
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            for (r, c) in target["cells"]:
                draw_cell_box(frame, r, c)
            p2_full = frame[
                DEFAULT_P2_REGION.y:DEFAULT_P2_REGION.y + DEFAULT_P2_REGION.height,
                DEFAULT_P2_REGION.x:DEFAULT_P2_REGION.x + DEFAULT_P2_REGION.width,
            ]
            t_sec = frame_idx / fps
            out_path = OUT_DIR / f"{target['name']}_f{frame_idx}_t{t_sec:.3f}_{label}.png"
            cv2.imwrite(str(out_path), p2_full)
            print(f"saved: {out_path}")
        cap.release()


if __name__ == "__main__":
    main()
