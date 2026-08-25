# -*- coding: utf-8 -*-
"""最終試合メディアン合成でも None になる動画のハミング距離診断 (2026-08-19)。

29 / c132 / 32 / 34 / 37 について、t_a (試合中) vs t_end (メディアン合成後)
の左右ハミング距離と、メディアンに使えたフレーム数を出す。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.match_winner import (  # noqa: E402
    LAST_END_MEDIAN_OFFSETS,
    MatchWinnerDetector,
    compare_digit_pairs,
    extract_digit_patches,
)

NEW_DIR = ROOT / "data" / "indicators_v2" / "boards_lean_subset50_2026-08-19"
FRAMES_DIR = ROOT / "data" / "frames"

# (tid, video, t_a=floor, t_end) 軽量検証 TSV から転記
CASES = [
    ("29", "video_29.mp4", 3163.2, 3220.266),
    ("c132", "video_c132.mp4", 2659.7, 2675.499),
    ("32", "video_32.mp4", 3365.6, 3434.5),
    ("34", "video_34.mp4", 2529.1, 2576.5),
    ("37", "video_37.mp4", 3366.2, 3435.5),
]


def main() -> None:
    det = MatchWinnerDetector.load_default()
    for tid, vf, t_a, t_end in CASES:
        cap = cv2.VideoCapture(str(FRAMES_DIR / vf))
        frame_a = det._read_frame(cap, t_a)
        la, ra = extract_digit_patches(frame_a)
        # メディアン合成の内訳
        n_used = 0
        for off in LAST_END_MEDIAN_OFFSETS:
            f = det._read_frame(cap, t_end + off)
            if f is not None and det._panel_detector.detect(f).present:
                n_used += 1
        lm, rm = det._median_digit_patches(cap, t_end, t_a)
        r = compare_digit_pairs(la, ra, lm, rm)
        # 単一フレーム比較 (参考)
        fb = det._read_frame(cap, t_end)
        lb, rb = extract_digit_patches(fb) if fb is not None else (None, None)
        r1 = compare_digit_pairs(la, ra, lb, rb)
        print(f"{tid}: median n={n_used} dl={r.left_hamming} dr={r.right_hamming} "
              f"winner={r.winner} | single dl={r1.left_hamming} dr={r1.right_hamming} "
              f"winner={r1.winner}")
        cap.release()


if __name__ == "__main__":
    main()
