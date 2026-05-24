"""matches.tsv に基づいて各試合の start/end の before/after 画像を書き出す。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2

TSV = Path("data/verify/match_boundaries_v4/video_02/matches.tsv")
VIDEO = Path("data/frames/video_02.mp4")
OUT = Path("data/verify/match_boundaries_v4/video_02")


def read_at(cap, t_sec: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080))
    return frame


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    lines = TSV.read_text(encoding="utf-8").splitlines()[1:]
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        idx, start, end, dur = parts[0], float(parts[1]), float(parts[2]), float(parts[3])

        # start 前後
        for offset, tag in [(-1.0, "start_before"), (0.0, "start_after")]:
            t = max(0.0, start + offset)
            f = read_at(cap, t)
            if f is not None:
                cv2.imwrite(str(OUT / f"m{int(idx):02d}_{tag}_{int(t):05d}s.png"), f)
        # end 前後
        for offset, tag in [(0.0, "end_before"), (1.0, "end_after")]:
            t = end + offset
            f = read_at(cap, t)
            if f is not None:
                cv2.imwrite(str(OUT / f"m{int(idx):02d}_{tag}_{int(t):05d}s.png"), f)
    cap.release()
    print(f"完了: {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
