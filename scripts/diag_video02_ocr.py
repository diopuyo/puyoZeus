"""video_02 (720p) で score OCR が動かない原因調査スクリプト。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from src.score_ocr import (
    DIGIT_HEIGHT,
    DIGIT_LEFTS_1P,
    DIGIT_TOP,
    DIGIT_WIDTH,
    EXPECTED_FRAME_SHAPE,
    SCORE_1P_REGION,
    SCORE_2P_REGION,
    ScoreOcr,
)


def main() -> int:
    out = Path("data/verify")
    out.mkdir(parents=True, exist_ok=True)

    with open("data/verify/match_boundaries_v4/video_02/matches.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    print(f"video_02 matches: {len(rows)}")
    print(f"first match: {rows[0]['start_sec']} - {rows[0]['end_sec']}")

    cap = cv2.VideoCapture("data/frames/video_02.mp4")
    print(f"video_02 raw: {int(cap.get(3))}x{int(cap.get(4))}")
    t = float(rows[0]["start_sec"]) + 30
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, raw = cap.read()
    print(f"raw frame at {t}s: shape={raw.shape}")

    # 1080p にリサイズ (INTER_AREA / INTER_CUBIC を比較)
    for interp_name, interp in [
        ("AREA", cv2.INTER_AREA),
        ("CUBIC", cv2.INTER_CUBIC),
        ("LANCZOS4", cv2.INTER_LANCZOS4),
    ]:
        resized = cv2.resize(raw, (EXPECTED_FRAME_SHAPE[1], EXPECTED_FRAME_SHAPE[0]),
                             interpolation=interp)
        y1, y2, x1, x2 = SCORE_1P_REGION
        roi = resized[y1:y2, x1:x2]
        cv2.imwrite(str(out / f"diag_video02_1p_roi_{interp_name}.png"), roi)

        # NCC を全 8 桁 × 全 10 クラスで計算
        all_scores: list[float] = []
        for i in range(8):
            cell = roi[DIGIT_TOP:DIGIT_TOP + DIGIT_HEIGHT,
                       DIGIT_LEFTS_1P[i]:DIGIT_LEFTS_1P[i] + DIGIT_WIDTH]
            gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            best = -1.0
            best_label = -1
            for n in range(10):
                tpl = cv2.imread(f"models/ui_templates/score_digits/digit_{n}.png")
                gray_tpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(gray_cell, gray_tpl,
                                        cv2.TM_CCOEFF_NORMED)
                m = float(res.max())
                if m > best:
                    best = m
                    best_label = n
            all_scores.append(best)
            print(f"  [{interp_name}] cell{i}: best digit={best_label} score={best:.3f}")
        print(f"  [{interp_name}] min NCC: {min(all_scores):.3f} max: {max(all_scores):.3f}")
        print()

    # 720p ネイティブの ROI を切ってみる (ROI を 720/1080 倍にスケール)
    scale = 720 / 1080
    y1, y2, x1, x2 = SCORE_1P_REGION
    yy1, yy2 = int(y1 * scale), int(y2 * scale)
    xx1, xx2 = int(x1 * scale), int(x2 * scale)
    roi_native = raw[yy1:yy2, xx1:xx2]
    print(f"720p native 1P ROI: shape={roi_native.shape}, mean={roi_native.mean():.1f}")
    cv2.imwrite(str(out / "diag_video02_1p_roi_native720.png"), roi_native)

    cap.release()
    print(f"\n出力: data/verify/diag_video02_1p_roi_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
