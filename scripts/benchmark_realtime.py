"""リアルタイム動作可能性のベンチマーク (= P2).

測定対象:
1. 個別 component の処理時間:
   - ImageReader.read_both_boards (= cell パッチ抽出 + 色分類)
   - CnnPatchClassifier 単体 (= 144 cells 一気にバッチ推論)
   - RecognitionPipeline.update (= 全 pipeline 1 frame)
2. CPU vs GPU 比較
3. 60fps 動画でリアルタイム動作可能か (= 16.67ms/frame 以下か)

使い方:
    PYTHONPATH=. python -m scripts.benchmark_realtime \\
        --video data/test_unknown/v91_match1_75s_720p.mp4 \\
        --n-frames 100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from src.recognition_pipeline import RecognitionPipeline


def time_it(fn, *args, n_trials: int = 1, **kwargs) -> tuple[float, Any]:
    """fn を n_trials 回実行し、 平均時間 (ms) と最後の戻り値を返す."""
    t0 = time.perf_counter()
    result = None
    for _ in range(n_trials):
        result = fn(*args, **kwargs)
    dt = (time.perf_counter() - t0) / n_trials
    return dt * 1000.0, result


def measure_pipeline(
    video_path: Path,
    n_frames: int,
    cnn_model_path: Path | None,
    label: str,
) -> dict:
    """RecognitionPipeline 1 frame 平均時間を測定."""
    print(f"\n[{label}] pipeline={video_path} model={cnn_model_path}")
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=cnn_model_path,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    print(f"[{label}] fps={fps:.1f}")

    # warmup (= 最初の 3 frame 除外)
    for _ in range(3):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080))
        pipeline.update(0, 0.0, frame)

    times: list[float] = []
    times_image_reader: list[float] = []
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080))
        # 個別測定: image_reader (= CNN 推論含む)
        t_ir0 = time.perf_counter()
        _b1, _b2 = pipeline._reader.read_both_boards(frame)
        t_ir = (time.perf_counter() - t_ir0) * 1000.0
        times_image_reader.append(t_ir)
        # 全 pipeline (= update)
        t0 = time.perf_counter()
        _ = pipeline.update(i + 4, (i + 4) / fps, frame)
        times.append((time.perf_counter() - t0) * 1000.0)
    cap.release()

    arr = np.array(times)
    arr_ir = np.array(times_image_reader)
    return {
        "label": label,
        "n_frames_measured": int(arr.size),
        "fps": fps,
        "pipeline_ms_mean": float(arr.mean()),
        "pipeline_ms_p50": float(np.percentile(arr, 50)),
        "pipeline_ms_p95": float(np.percentile(arr, 95)),
        "pipeline_ms_max": float(arr.max()),
        "imagereader_ms_mean": float(arr_ir.mean()),
        "imagereader_ms_p95": float(np.percentile(arr_ir, 95)),
        "realtime_at_60fps": bool(arr.mean() <= 1000.0 / 60.0),
        "realtime_at_30fps": bool(arr.mean() <= 1000.0 / 30.0),
        "realtime_at_15fps": bool(arr.mean() <= 1000.0 / 15.0),
        "effective_fps": float(1000.0 / arr.mean()) if arr.size > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", type=Path,
        default=Path("data/test_unknown/v91_match1_75s_720p.mp4"),
    )
    parser.add_argument("--n-frames", type=int, default=100)
    parser.add_argument(
        "--cnn-model", type=Path,
        default=Path("models/cnn_phase_b_finetuned.pt"),
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"[error] video not found: {args.video}")
        return 1

    print("=" * 60)
    print(f"benchmark: {args.video.name}, n_frames={args.n_frames}")
    print("=" * 60)

    # 構成 1: HSV-only (= cnn_model=None)
    r_hsv = measure_pipeline(
        args.video, args.n_frames, None, label="HSV-only (no CNN)",
    )
    # 構成 2: HSV + CNN (= fine-tuned model)
    r_cnn = measure_pipeline(
        args.video, args.n_frames, args.cnn_model, label="HSV+CNN finetuned",
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    for r in (r_hsv, r_cnn):
        print(f"\n--- {r['label']} ---")
        print(f"  n_frames     : {r['n_frames_measured']}")
        print(f"  pipeline ms  : mean={r['pipeline_ms_mean']:.2f}  "
              f"p50={r['pipeline_ms_p50']:.2f}  "
              f"p95={r['pipeline_ms_p95']:.2f}  max={r['pipeline_ms_max']:.2f}")
        print(f"  imagereader  : mean={r['imagereader_ms_mean']:.2f}  "
              f"p95={r['imagereader_ms_p95']:.2f}")
        print(f"  effective fps: {r['effective_fps']:.1f}")
        print(
            f"  realtime?    : 60fps={r['realtime_at_60fps']}  "
            f"30fps={r['realtime_at_30fps']}  15fps={r['realtime_at_15fps']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
