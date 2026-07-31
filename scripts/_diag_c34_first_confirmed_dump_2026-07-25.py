"""補助診断: video_c34 game1 開始直後、confirmed_board が初めて非None になった
瞬間のセル内容をダンプする (2026-07-25、read-only)。

_diag_c34_reflection_lag_2026-07-25.py の初期盤面ダンプは game_start_sec 直前・
直後 (t=465.6/465.63) のみで、その時点では両者とも confirmed=None だった
(cnn_board もほぼ空)。ユーザーが目視した「青2個が既にある」瞬間を特定する
ため、game_start_sec 以降で confirmed_board が最初に非None になったフレームの
グリッドをダンプする。

Usage:
    PYTHONPATH=. ./venv/bin/python scripts/_diag_c34_first_confirmed_dump_2026-07-25.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2

for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO_STEM: str = "c34"
START_SEC: float = 460.0
MAX_SEC: float = 15.0
GAME_START_SEC: float = 465.6


def main() -> None:
    cv2.setNumThreads(1)
    video_path = PROJ_ROOT / "data" / "frames" / f"video_{VIDEO_STEM}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(START_SEC * fps)
    end_frame = int((START_SEC + MAX_SEC) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    pipe = RecognitionPipeline.load_default(enable_landing_observed_color=True)
    pipe.set_video_id(VIDEO_STEM)

    found_1p = False
    found_2p = False
    fi = start_frame
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t = fi / fps
        r = pipe.update(fi, t, frame)
        if t >= GAME_START_SEC and not found_1p and r.p1.confirmed_board is not None:
            found_1p = True
            print(f"[1P] t={t:.3f}s state={r.p1.state.name} board_none_reason(prev)=n/a")
            print(r.p1.confirmed_board)
            print(f"[1P] cnn_board同時刻:\n{r.p1.cnn_board}")
        if t >= GAME_START_SEC and not found_2p and r.p2.confirmed_board is not None:
            found_2p = True
            print(f"[2P] t={t:.3f}s state={r.p2.state.name}")
            print(r.p2.confirmed_board)
            print(f"[2P] cnn_board同時刻:\n{r.p2.cnn_board}")
        if found_1p and found_2p:
            break
        fi += 1
    cap.release()


if __name__ == "__main__":
    main()
