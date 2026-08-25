"""問1/問2 証拠フレーム抽出 (計装スクリプト、本体コード非変更)。

c109 の実際の試合境界フレーム (t=3626-3637, t=3698-3703, t=3787-3792) と、
「単一巨大連鎖」区間 (t=3660-3693) の実画面を保存する。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

OUT_DIR = Path("data/verify/diag_boundary_2026-08-17/frames")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO = Path("data/frames/video_c109.mp4")

# (label, t_sec)
TARGETS = [
    ("boundary1_before", 3625.0),
    ("boundary1_during", 3631.0),
    ("boundary1_after", 3637.0),
    ("chain_mid_9356", 3668.5),
    ("chain_mid_after_fire", 3684.0),
    ("boundary2_before", 3691.5),
    ("boundary2_during", 3699.0),
    ("boundary2_after", 3703.0),
]


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print("OPEN FAILED")
        return 1
    for label, t_sec in TARGETS:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[{label}] READ FAILED at t={t_sec}")
            continue
        out_path = OUT_DIR / f"c109_{label}_t{t_sec:.1f}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"[{label}] saved -> {out_path}")
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
