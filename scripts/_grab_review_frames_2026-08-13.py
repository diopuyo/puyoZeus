"""userレビュー指摘場面の実画面フレームを抽出する (read-only, 2026-08-13).

data/verify/demo_review_2026-08-13/frames/ に保存する (レビュー規約
feedback_review_actual_screen_frames_2026-07-24 準拠)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT_DIR = Path("data/verify/demo_review_2026-08-13/frames")
# 抽出対象の source 秒 (task 記載の指摘場面)
TARGET_SECS = [187.5, 188.0, 188.5, 189.0, 189.5,
               195.5, 196.0, 196.5, 197.0, 197.5, 198.0, 198.5, 199.0,
               # 場面3: 4試合目序盤 (source t=311-335) の設置確定遅延指摘
               309.0, 311.0, 313.0, 315.0, 320.0, 325.0, 330.0, 335.0,
               # 場面4: 5試合目 (source t=370-410) の連鎖後残像指摘
               396.5, 397.5, 398.0, 402.0, 405.0, 407.0, 407.8, 408.5, 409.5]


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"cannot open {VIDEO}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for t in TARGET_SECS:
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[skip] t={t} frame_idx={frame_idx} 読み込み失敗", file=sys.stderr)
            continue
        out_path = OUT_DIR / f"source_t{t:07.2f}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"[saved] {out_path}")
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
