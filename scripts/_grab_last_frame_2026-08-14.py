"""デモ2の最終フレーム確認 (突然の切断か否か、read-only)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO = Path("data/verify/demo_fixed_2026-08-13/demo2_video74_3match.mp4")
OUT_DIR = Path("data/verify/demo_fixed_2026-08-13/frames_demo2")


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"total_frames={total} fps={fps} dur={total / fps:.2f}s")
    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ok, frame = cap.read()
    if ok:
        out_path = OUT_DIR / "demo2_last_frame.png"
        cv2.imwrite(str(out_path), frame)
        print(f"[saved] {out_path}")
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
