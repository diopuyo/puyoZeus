"""色→空 HSV 照合ガード 既定ON化の速度影響 計測 (read-only, 2026-07-30)。

enable_puyo_to_empty_hsv_guard の既定 ON 化により read_board_hsv_only が
NON-STABLE フレームでも毎フレーム走るようになる。その 1 フレーム所要時間と
HSV 計算が占める割合を OFF/ON で実測する。

- src/ は変更しない。read_board_hsv_only を計時用にラップ (元へ必ず復元)。
- 同一動画・同一窓を OFF/ON で 2 回走らせ、pipe.update の総時間と
  read_board_hsv_only の総時間・呼出回数を比較する。

Usage (WSL):
    PYTHONPATH=. nice -n 19 ./venv/bin/python \
        scripts/_diag_hsv_guard_speed_2026-07-30.py --video c10 \
        --start-sec 60 --max-sec 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import cv2  # noqa: E402

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.recognition_pipeline import RecognitionPipeline  # noqa: E402


def _run(video: str, start_sec: float, max_sec: float, guard: bool) -> dict:
    """1 構成 (guard on/off) で走査し所要時間統計を返す."""
    cv2.setNumThreads(1)
    path = PROJ_ROOT / "data" / "frames" / f"video_{video}.mp4"
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_f = int(start_sec * fps)
    end_f = int((start_sec + max_sec) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_f))

    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
        enable_puyo_to_empty_hsv_guard=guard,
    )
    pipe.set_video_id(video)

    # read_board_hsv_only を計時ラップ (呼出回数・累積秒)。
    reader = pipe._reader
    orig = reader.read_board_hsv_only
    stats = {"hsv_calls": 0, "hsv_sec": 0.0}

    def _wrapped(frame_bgr, region):  # type: ignore[no-untyped-def]
        t = time.perf_counter()
        r = orig(frame_bgr, region)
        stats["hsv_sec"] += time.perf_counter() - t
        stats["hsv_calls"] += 1
        return r

    reader.read_board_hsv_only = _wrapped  # type: ignore[assignment]
    try:
        n = 0
        t_update = 0.0
        fi = start_f
        while fi < end_f:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            t = fi / fps
            t0 = time.perf_counter()
            pipe.update(fi, t, frame)
            t_update += time.perf_counter() - t0
            fi += 1
            n += 1
    finally:
        reader.read_board_hsv_only = orig  # type: ignore[assignment]
        cap.release()

    return {
        "guard": guard, "frames": n, "fps_src": fps,
        "update_sec": t_update,
        "update_ms_per_frame": 1000.0 * t_update / n if n else 0.0,
        "eff_fps": n / t_update if t_update else 0.0,
        "hsv_calls": stats["hsv_calls"],
        "hsv_sec": stats["hsv_sec"],
        "hsv_ms_per_frame": 1000.0 * stats["hsv_sec"] / n if n else 0.0,
        "hsv_pct_of_update": 100.0 * stats["hsv_sec"] / t_update if t_update else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="c10")
    ap.add_argument("--start-sec", type=float, default=60.0)
    ap.add_argument("--max-sec", type=float, default=30.0)
    a = ap.parse_args()
    print(f"[{time.strftime('%H:%M:%S')}] video={a.video} "
          f"start={a.start_sec} dur={a.max_sec}", flush=True)
    off = _run(a.video, a.start_sec, a.max_sec, guard=False)
    print(f"[OFF] {off}", flush=True)
    on = _run(a.video, a.start_sec, a.max_sec, guard=True)
    print(f"[ON ] {on}", flush=True)
    slow = (on["update_ms_per_frame"] - off["update_ms_per_frame"])
    pct = 100.0 * slow / off["update_ms_per_frame"] if off["update_ms_per_frame"] else 0.0
    print(f"[DIFF] 1フレーム所要 OFF={off['update_ms_per_frame']:.2f}ms "
          f"ON={on['update_ms_per_frame']:.2f}ms "
          f"増加={slow:+.2f}ms ({pct:+.1f}%) | "
          f"HSV呼出 OFF={off['hsv_calls']} ON={on['hsv_calls']} | "
          f"HSVがupdate内で占める割合 OFF={off['hsv_pct_of_update']:.1f}% "
          f"ON={on['hsv_pct_of_update']:.1f}%", flush=True)


if __name__ == "__main__":
    main()
