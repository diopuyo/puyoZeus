"""テンプレNCC最高マッチフレームを検索する一時スクリプト"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")
from src.ojama_warning import (
    P1_BOARD_X,
    P2_BOARD_X,
    WARNING_BOTTOM_Y,
    WARNING_TOP_Y,
    CELL_WIDTH,
    ICON_CROP_HALF,
    TEMPLATE_DIR_DEFAULT,
    _load_templates,
    _ncc_score,
)

VIDEO_PATH = "data/frames/video_124_4min.mp4"


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    templates = _load_templates(TEMPLATE_DIR_DEFAULT)
    best: dict[str, dict] = {k: {"t": 0, "side": "", "cell": 0, "score": 0.0} for k in templates}

    max_t = int(total / fps) - 5
    for t_sec in range(5, max_t, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
        ret, frame = cap.read()
        if not ret:
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080))

        for board_x, side in [(P1_BOARD_X, "1P"), (P2_BOARD_X, "2P")]:
            for i in range(6):
                cx = board_x + int((i + 0.5) * CELL_WIDTH)
                x1 = max(0, cx - ICON_CROP_HALF)
                x2 = min(1920, cx + ICON_CROP_HALF)
                cell = frame[WARNING_TOP_Y:WARNING_BOTTOM_Y, x1:x2]

                for kind in list(templates.keys()):
                    try:
                        score = max(_ncc_score(cell, timg) for timg in templates[kind])
                        if score > best[kind]["score"]:
                            best[kind] = {"t": t_sec, "side": side, "cell": i, "score": score}
                    except Exception:
                        pass

    cap.release()
    print("Best NCC match per template:")
    for kind, b in sorted(best.items(), key=lambda x: x[1]["score"], reverse=True):
        print(f"  {kind}: t={b['t']}s {b['side']} cell[{b['cell']}] NCC={b['score']:.3f}")


if __name__ == "__main__":
    main()
