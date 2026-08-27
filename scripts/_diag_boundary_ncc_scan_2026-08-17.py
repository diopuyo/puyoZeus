"""問1計装: MatchEndDetector / ScoreZeroDetector の生NCCスコアを
複数動画で全編スキャンし、動画別の検出率を計測する (本体コード変更なし)。

c109 (W20/W21実例) だけでなく c13 / c96 でもクロスビデオ精度を測る。
5秒間隔の粗いサンプリングで全編を走査し、各時刻での
  - score_zero: s1 (1P NCC), s2 (2P NCC)
  - match_end: best_score, best_name (yatta/batan どちらに最も近いか)
を記録する。閾値 (score_zero=0.85, match_end=0.55) に対する生値の分布を見る。

出力: data/verify/diag_boundary_2026-08-17/ncc_scan_<video>.csv
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from src.match_end_detector import MatchEndDetector  # noqa: E402
from src.score_zero import ScoreZeroDetector  # noqa: E402

OUT_DIR = Path("data/verify/diag_boundary_2026-08-17")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_INTERVAL_SEC = 5.0

VIDEOS = {
    "c109": Path("data/frames/video_c109.mp4"),
    "c13": Path("data/frames/video_c13.mp4"),
    "c96": Path("data/frames/video_c96.mp4"),
}


def scan_video(name: str, path: Path) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[{name}] OPEN FAILED: {path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur_sec = n_frames / fps if fps > 0 else 0.0
    print(f"[{name}] fps={fps} n_frames={n_frames} dur_sec={dur_sec:.1f}", flush=True)

    score_zero_det = ScoreZeroDetector.load_default()
    match_end_det = MatchEndDetector.load_default()

    out_csv = OUT_DIR / f"ncc_scan_{name}.csv"
    t0 = time.time()
    n_written = 0
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video", "t_sec", "sz_s1", "sz_s2", "sz_both_zero",
            "me_best_score", "me_best_name", "me_detected",
        ])
        t_sec = 0.0
        while t_sec < dur_sec:
            cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                t_sec += SAMPLE_INTERVAL_SEC
                continue
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080))
            sz = score_zero_det.detect(frame)
            me = match_end_det.detect(frame)
            writer.writerow([
                name, round(t_sec, 2), round(sz.score_1p, 4), round(sz.score_2p, 4),
                int(sz.both_zero), round(me.score, 4), me.template_name or "",
                int(me.detected),
            ])
            n_written += 1
            t_sec += SAMPLE_INTERVAL_SEC
    dt = time.time() - t0
    print(f"[{name}] wrote {n_written} rows in {dt:.1f}s -> {out_csv}", flush=True)
    cap.release()


def main() -> int:
    for name, path in VIDEOS.items():
        if not path.exists():
            print(f"[{name}] MISSING: {path}")
            continue
        scan_video(name, path)
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
