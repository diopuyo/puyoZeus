"""デモ2 t=28-50秒を1秒間隔でコンタクトシート化 (両者発火ホールド#9の精査用、read-only)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO = Path("data/verify/demo_fixed_2026-08-13/demo2_video74_3match.mp4")
OUT_DIR = Path("data/verify/demo_fixed_2026-08-13/frames_demo2/contact_sheets")

CROP = (1420, 20, 1920, 480)
T_START = 28
T_END = 50


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"cannot open {VIDEO}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    strips = []
    for t in range(T_START, T_END + 1):
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        x1, y1, x2, y2 = CROP
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (500, 230))
        labeled = np.zeros((230, 620, 3), dtype=np.uint8)
        labeled[:, 120:] = crop
        cv2.putText(labeled, f"t={t}s", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)
        strips.append(labeled)
    cap.release()

    sheet = np.vstack(strips)
    out_path = OUT_DIR / "sheet_fine_28to50.png"
    cv2.imwrite(str(out_path), sheet)
    print(f"[saved] {out_path} ({len(strips)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
