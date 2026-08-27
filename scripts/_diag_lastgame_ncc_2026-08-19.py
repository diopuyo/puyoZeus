# -*- coding: utf-8 -*-
"""最終試合の終点判別: NCC (輝度不変) の判別力実測 (2026-08-19)。

勝利演出のパネル発光で二値指紋ハミングが両側とも大きくなる問題に対し、
グレースケール NCC (TM_CCOEFF_NORMED、平均差し引き=輝度・コントラスト
シフトに頑健) が「変わっていない側 (敗者) = 高 NCC / 変わった側 (勝者) =
低 NCC」を分離できるかを、真値が分かる 11 本で測る。

多試合スパン (35/38/39: npz最終試合の後も動画内で試合が続き両側増分) では
両側とも低 NCC になり判定不能のままであるべき (安全側の確認)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.match_winner import (  # noqa: E402
    MatchWinnerDetector,
    extract_digit_patches,
)

FRAMES_DIR = ROOT / "data" / "frames"

# (tid, video, t_a, t_end, 期待勝者[score系統/実画面より], 備考)
CASES = [
    ("29", "video_29.mp4", 3163.2, 3220.266, "1P", "実画面確認済 29→30"),
    ("c132", "video_c132.mp4", 2659.7, 2675.499, "1P", "実画面確認済 29→30"),
    ("32", "video_32.mp4", 3365.6, 3434.5, "1P", "score系統"),
    ("34", "video_34.mp4", 2529.1, 2576.5, "1P", "score系統"),
    ("37", "video_37.mp4", 3366.2, 3435.5, "2P", "score系統"),
    ("31", "video_31.mp4", 3139.6, 3193.5, "2P", "既にhamming判定OK"),
    ("33", "video_33.mp4", 3216.7, 3283.5, "1P", "既にhamming判定OK"),
    ("c109", "video_c109.mp4", 11279.9, 11364.2, "2P", "既にhamming判定OK"),
    ("35", "video_35.mp4", 3117.3, 3590.533, None, "多試合スパン→判定不能が正"),
    ("38", "video_38.mp4", 1528.6, 3214.5, None, "多試合スパン→判定不能が正"),
    ("39", "video_39.mp4", 1106.5, 3030.499, None, "多試合スパン→判定不能が正"),
]


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    v = cv2.matchTemplate(ga, gb, cv2.TM_CCOEFF_NORMED)
    return float(v[0, 0])


def main() -> None:
    det = MatchWinnerDetector.load_default()
    for tid, vf, t_a, t_end, expect, note in CASES:
        cap = cv2.VideoCapture(str(FRAMES_DIR / vf))
        frame_a = det._read_frame(cap, t_a)
        la, ra = extract_digit_patches(frame_a)
        lm, rm = det._median_digit_patches(cap, t_end, t_a)
        cap.release()
        if la is None or lm is None:
            print(f"{tid}: patch取得失敗")
            continue
        nl = _ncc(la, lm)
        nr = _ncc(ra, rm)
        pred = None
        # 仮ルール: 低い側=勝者。高い側>=0.55 (静止確認) かつ差>=0.20
        if max(nl, nr) >= 0.55 and abs(nl - nr) >= 0.20:
            pred = "1P" if nl < nr else "2P"
        ok = "OK" if pred == expect else ("--" if pred is None and expect is None else "NG")
        print(f"{tid}: ncc_L={nl:.3f} ncc_R={nr:.3f} pred={pred} expect={expect} "
              f"[{ok}] {note}")


if __name__ == "__main__":
    main()
