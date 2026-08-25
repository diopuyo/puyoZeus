"""問1証拠フレーム追加抽出: 3788-3790秒の score_zero 高速フリッカ区間。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

OUT_DIR = Path("data/verify/diag_boundary_2026-08-17/frames")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO = Path("data/frames/video_c109.mp4")

TARGETS = [
    ("flicker_3787_5", 3787.5),
    ("flicker_3788_1", 3788.1),
    ("flicker_3788_9", 3788.9),
    ("flicker_3789_8", 3789.8),
    ("flicker_3790_5", 3790.5),
]


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    for label, t_sec in TARGETS:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        out_path = OUT_DIR / f"c109_{label}_t{t_sec:.2f}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
