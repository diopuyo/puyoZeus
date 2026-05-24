"""TelopDetector のスコア検証 (表示中/非表示時の NCC 値分離確認)。

m27 (1636-1713s) を 1 秒間隔でテンプレートマッチして時系列スコアを出す。
試合開始直後 (1640s 等) は非表示、試合終了告知 (1670s 以降) は表示中で
スコアが分離するはず。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.console_init import init_console  # noqa: E402
init_console()

import cv2

from src.telop_detector import TelopDetector

VIDEO = "data/frames/video_01.mp4"
TIMES = list(range(1636, 1714))  # 1 秒刻み


def main() -> int:
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"video open failed: {VIDEO}")
        return 1

    detector = TelopDetector.load_default()
    print(f"templates loaded: {list(detector._templates.keys())}")

    for t in TIMES:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        if fr.shape[:2] != (1080, 1920):
            fr = cv2.resize(fr, (1920, 1080), interpolation=cv2.INTER_AREA)
        res = detector.detect(fr)
        marker = "★" if res.is_visible else " "
        print(f"  t={t:4d}s {marker} score={res.score:.3f} "
              f"tmpl={res.template_name}")

    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
