"""Phase I: 動画から擬似ラベルを収集して LabelStore に永続化.

Usage:
    python scripts/phase_i_collect_pseudo_labels.py \
        --video data/frames/video_02.mp4 \
        --video-id video_02 \
        --max-frames 6000 \
        [--cnn-model models/cnn_global_best.pt]

出力:
    data/pseudo_labels/{video_id}/{component}.jsonl

設計:
    - RecognitionPipeline(enable_pseudo_label=True) で動画 1 本を流す
    - 60 frame ごとに flush_pseudo_labels() で disk に書き出し
    - frame サンプリングは BOARD_INTERVAL_SEC=0.2s (5 fps 相当)
    - 720p 動画は 1080p にリサイズしてから投入
      (next_detector / score_validator / next_validator が 1080p 必須)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.recognition_pipeline import RecognitionPipeline
from src.sampling_config import BOARD_INTERVAL_SEC
from src.self_supervised.label_store import LabelStore


# flush 間隔 (frame 数)
FLUSH_EVERY_N_FRAMES: int = 60

# pipeline は 1080p (1920x1080) ROI 前提なので、入力動画が異なる解像度なら
# ここでリサイズしてから渡す。
TARGET_FRAME_SIZE: tuple[int, int] = (1920, 1080)


def _ensure_target_size(frame):
    """frame を TARGET_FRAME_SIZE に揃える. 既に同サイズなら no-op."""
    h, w = frame.shape[:2]
    tw, th = TARGET_FRAME_SIZE
    if (w, h) == (tw, th):
        return frame
    return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path,
                         help="入力動画ファイル")
    parser.add_argument("--video-id", required=True, type=str,
                         help="ストア内の video_id")
    parser.add_argument("--max-frames", type=int, default=0,
                         help="最大処理 frame 数 (0=全部)")
    parser.add_argument("--cnn-model", type=Path,
                         default=Path("models/cnn_global_best.pt"),
                         help="CNN model path")
    parser.add_argument("--store-root", type=Path,
                         default=Path("data/pseudo_labels"),
                         help="LabelStore ルート")
    parser.add_argument(
        "--sampling-interval-sec", type=float,
        default=BOARD_INTERVAL_SEC,
        help="frame サンプリング間隔 (秒)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    cnn_path = args.cnn_model if args.cnn_model.is_file() else None
    store = LabelStore(video_id=args.video_id, root=args.store_root)
    pipe = RecognitionPipeline.load_default(
        cnn_model_path=cnn_path,
        load_score_ocr=True,
        enable_chain_tracker=True,
        load_next_detector=True,
        enable_pseudo_label=True,
        pseudo_label_store=store,
    )
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sampling_step_frame = max(1, int(round(args.sampling_interval_sec * fps)))
    print(
        f"[phase_i] video={args.video.name} fps={fps:.2f} "
        f"step={sampling_step_frame} frames",
    )
    t_start = time.time()
    frame_idx = 0
    processed = 0
    flushed_total = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % sampling_step_frame == 0:
                t_sec = frame_idx / fps
                # 入力動画が 1080p 以外でも pipeline へは 1080p で渡す
                frame_norm = _ensure_target_size(frame)
                pipe.update(processed, t_sec, frame_norm)
                processed += 1
                if processed % FLUSH_EVERY_N_FRAMES == 0:
                    n = pipe.flush_pseudo_labels()
                    flushed_total += n
                    if n > 0:
                        print(
                            f"[phase_i] frame={frame_idx} "
                            f"flushed={n} total={flushed_total}",
                        )
                if args.max_frames > 0 and processed >= args.max_frames:
                    break
            frame_idx += 1
    finally:
        # 残り flush
        flushed_total += pipe.flush_pseudo_labels()
        cap.release()
    elapsed = time.time() - t_start
    print(
        f"[phase_i] done video_id={args.video_id} "
        f"frames={processed} flushed={flushed_total} elapsed={elapsed:.1f}s",
    )
    print(f"[phase_i] stats: {store.stats()}")


if __name__ == "__main__":
    main()
