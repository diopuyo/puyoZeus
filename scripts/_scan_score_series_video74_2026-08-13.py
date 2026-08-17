"""demo2用: video_74.mp4 のスコアOCRを粗くスキャンし、試合境界(0-0リセット)を探す。
一時利用スクリプト (使い捨て、命名規則 _xxx_YYYY-MM-DD.py に従う)。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2

from src.score_ocr import ScoreOcr

VIDEO_PATH = _ROOT / "data" / "frames" / "video_74.mp4"
STEP_SEC = float(os.environ.get("SCAN_STEP_SEC", "2.0"))
START_SEC = float(os.environ.get("SCAN_START_SEC", "0.0"))
END_SEC = float(os.environ.get("SCAN_END_SEC", "900.0"))


def main() -> int:
    ocr = ScoreOcr.load_default()
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[error] open failed: {VIDEO_PATH}")
        return 1

    t = START_SEC
    prev_1p: int | None = None
    prev_2p: int | None = None
    while t <= END_SEC:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            t += STEP_SEC
            continue
        res = ocr.read(frame)
        flag = ""
        if res.score_1p is not None and res.score_2p is not None:
            if prev_1p is not None and (res.score_1p < prev_1p or res.score_2p < prev_2p):
                flag = " <<< RESET candidate"
            prev_1p, prev_2p = res.score_1p, res.score_2p
        print(f"t={t:7.1f} 1p={res.score_1p} (c={res.confidence_1p:.2f}) "
              f"2p={res.score_2p} (c={res.confidence_2p:.2f}){flag}")
        t += STEP_SEC
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
