"""認識パイプラインの素の fps を測る (プロファイラ無し)。

cProfile を使うと Python フレームのオーバーヘッドで絶対値が 2 倍以上膨らむため、
「今何 fps なのか」は計測器を挟まずに測る必要がある。

2026-07-30 の基準値 (4.4fps → 8.07fps) は video_c60 t=1451、60フレーム、
threads=16、収集ジョブ競合下で測られている。同条件で比較するため
既定値をそれに合わせている。

出力は 1 フレームあたり ms と fps。30fps 予算 33.3ms に対する差も出す。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_fps_bench_2026-07-30 \
        --video data/frames/video_c60.mp4 --start-sec 1451 --frames 60
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

TARGET_W, TARGET_H = 1920, 1080
# 30fps / 20fps の 1 フレーム予算 [ms]
BUDGET_30FPS_MS: float = 1000.0 / 30.0
BUDGET_20FPS_MS: float = 1000.0 / 20.0
# 立ち上がり (キャッシュ構築・初回 CNN ロード) を除外するフレーム数
WARMUP_FRAMES: int = 10


def _read_frames(video: Path, frames: int, start_sec: float) -> list[np.ndarray]:
    """動画から連続フレームを読み出す (1920x1080 に正規化)。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けない: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
    out: list[np.ndarray] = []
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != TARGET_W or frame.shape[0] != TARGET_H:
            frame = cv2.resize(
                frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA,
            )
        out.append(frame)
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=Path("data/frames/video_c60.mp4"))
    ap.add_argument("--start-sec", type=float, default=1451.0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument(
        "--cv-threads", type=int, default=-1,
        help="cv2.setNumThreads の値。-1 で既定 (基準測定と同じ)",
    )
    ap.add_argument(
        "--large-roi-throttle", action="store_true",
        help="大 ROI 走査の間引きを有効化して測る",
    )
    ap.add_argument(
        "--repeat", type=int, default=1,
        help="同一フレーム列を何回まわすか (競合ノイズ対策、中央値を取る)",
    )
    args = ap.parse_args()

    if args.cv_threads >= 0:
        cv2.setNumThreads(args.cv_threads)

    from src.recognition_pipeline import RecognitionPipeline

    frames = _read_frames(args.video, args.frames, args.start_sec)
    if not frames:
        raise RuntimeError("フレームを読めなかった")
    # 間引きの実効値を表示する (2026-07-31 に既定 ON 化したので、
    # 「フラグ未指定 = OFF」と表示すると誤読する)
    import inspect

    lib_default = inspect.signature(
        RecognitionPipeline.load_default,
    ).parameters["enable_large_roi_throttle"].default
    effective = True if args.large_roi_throttle else bool(lib_default)
    print(
        f"動画: {args.video.name}  t={args.start_sec}s  "
        f"フレーム数: {len(frames)}  cv_threads={cv2.getNumThreads()}  "
        f"間引き={'ON' if effective else 'OFF'}"
        f" (ライブラリ既定={lib_default}, 明示指定={args.large_roi_throttle})"
    )

    # 旧コード (d7fc6c2) との A/B 用: 旧版は enable_large_roi_throttle を知らないので
    # フラグ未指定時は引数を渡さない (TypeError 回避)。
    kwargs = (
        {"enable_large_roi_throttle": True} if args.large_roi_throttle else {}
    )
    # 複数回まわして中央値を取る (競合下の分散が ±4% あるため 1 回では判定不能)
    all_steady: list[np.ndarray] = []
    for rep in range(args.repeat):
        pipe = RecognitionPipeline.load_default(**kwargs)
        per_frame: list[float] = []
        for idx, frame in enumerate(frames):
            t0 = time.perf_counter()
            pipe.update(idx, idx / 30.0, frame)
            per_frame.append((time.perf_counter() - t0) * 1000.0)
        rep_arr = np.asarray(per_frame)
        rep_steady = (
            rep_arr[WARMUP_FRAMES:] if rep_arr.size > WARMUP_FRAMES else rep_arr
        )
        all_steady.append(rep_steady)
        print(
            f"  試行{rep + 1}: 定常中央 {float(np.median(rep_steady)):.1f}ms "
            f"({1000.0 / float(np.median(rep_steady)):.2f} fps)"
        )

    arr = np.concatenate(all_steady)
    steady = arr
    # 競合下では平均が外れ値に引きずられるので中央値を主指標にする
    mean_ms = float(np.median(steady))
    print(
        f"\n定常 ({WARMUP_FRAMES}フレーム以降を {args.repeat} 試行分結合 "
        f"n={steady.size}): 中央 {mean_ms:.1f}ms  平均 {float(steady.mean()):.1f}ms  "
        f"p90 {float(np.percentile(steady, 90)):.1f}ms"
    )
    print(f"→ {1000.0 / mean_ms:.2f} fps 相当 (中央値ベース)")
    print(
        f"\n20fps 予算 {BUDGET_20FPS_MS:.1f}ms: "
        f"{'達成' if mean_ms <= BUDGET_20FPS_MS else f'あと {mean_ms - BUDGET_20FPS_MS:.1f}ms 削減'}"
    )
    print(
        f"30fps 予算 {BUDGET_30FPS_MS:.1f}ms: "
        f"{'達成' if mean_ms <= BUDGET_30FPS_MS else f'あと {mean_ms - BUDGET_30FPS_MS:.1f}ms 削減'}"
    )


if __name__ == "__main__":
    main()
