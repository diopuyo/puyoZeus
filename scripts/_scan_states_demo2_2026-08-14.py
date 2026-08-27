"""デモ2の状態ラベル (1P状態/2P状態) を全域2秒間隔でコンタクトシート化 (read-only, 2026-08-14).

両者同時CHAIN (#9ホールド対象) の場面探索用。1フレームスポット確認でなく全域を
機械的に並べてuser/検収レビュアが素早く目視できるようにする。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO = Path("data/verify/demo_fixed_2026-08-13/demo2_video74_3match.mp4")
OUT_DIR = Path("data/verify/demo_fixed_2026-08-13/frames_demo2/contact_sheets")

# 状態ラベル+有利不利%の領域 (右パネル)
CROP = (1420, 20, 1920, 480)
STEP_SEC = 2
DURATION_SEC = 181
ROWS_PER_SHEET = 18


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"cannot open {VIDEO}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    times = list(range(0, DURATION_SEC, STEP_SEC))
    strips = []
    for t in times:
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        x1, y1, x2, y2 = CROP
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (500, 230))
        # 左に時刻ラベルを追加
        labeled = np.zeros((230, 620, 3), dtype=np.uint8)
        labeled[:, 120:] = crop
        cv2.putText(labeled, f"t={t}s", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)
        strips.append(labeled)
    cap.release()

    n_sheets = (len(strips) + ROWS_PER_SHEET - 1) // ROWS_PER_SHEET
    for si in range(n_sheets):
        chunk = strips[si * ROWS_PER_SHEET:(si + 1) * ROWS_PER_SHEET]
        sheet = np.vstack(chunk)
        out_path = OUT_DIR / f"sheet_{si:02d}.png"
        cv2.imwrite(str(out_path), sheet)
        print(f"[saved] {out_path} ({len(chunk)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
