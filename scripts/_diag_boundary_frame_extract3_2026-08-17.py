"""問1証拠フレーム追加抽出: t=3632.3付近の瞬間的リンス(スパイク)区間。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

OUT_DIR = Path("data/verify/diag_boundary_2026-08-17/frames")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO = Path("data/frames/video_c109.mp4")

TARGETS = [
    ("blip_3632_1", 3632.1),
    ("blip_3632_33", 3632.33),
    ("blip_3632_5", 3632.5),
    ("blip2_3698_6", 3698.6),
    ("blip2_3698_8", 3698.8),
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
