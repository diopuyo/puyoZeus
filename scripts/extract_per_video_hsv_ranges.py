"""動画別 HSV ranges DB 構築 (Phase I.c 段階 3、 memory `realtime_hsv` 段階 3)。

動画 1 本を再生して OnlineHsvCalibrator を学習させ、 学習結果 (HSV ranges)
を JSON ファイルに save する。次回起動時に load して initial として inject
すれば、 動画再生開始から即時に動画別最適化された認識精度を得られる。

使い方:
    python scripts/extract_per_video_hsv_ranges.py \
        --video data/frames/video_29.mp4 \
        --video-id v29 \
        --cnn-model models/cnn_phase_b_finetuned.pt \
        --out data/per_video_hsv_ranges/v29.json \
        [--max-frames 3000]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from src.recognition_pipeline import RecognitionPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--video-id", type=str, required=True)
    p.add_argument("--cnn-model", type=Path,
                    default=Path("models/cnn_phase_b_finetuned.pt"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=3000,
                    help="処理 frame 数上限 (1500 frame ≈ 50 秒で十分採取)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pipe = RecognitionPipeline.load_default(
        cnn_model_path=args.cnn_model,
        force_in_match=True,
    )
    if pipe._online_hsv is None:
        raise RuntimeError("online_hsv not initialized (CNN model required)")
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_target = min(n_total, int(args.max_frames))
    print(
        f"[hsv_db] video={args.video} id={args.video_id} "
        f"fps={fps:.1f} target={n_target}",
    )
    t0 = time.time()
    for fi in range(n_target):
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        if fr.shape[:2] != (1080, 1920):
            fr = cv2.resize(fr, (1920, 1080), interpolation=cv2.INTER_AREA)
        pipe.update(fi, fi / fps, fr)
        if fi > 0 and fi % 500 == 0:
            counts = pipe._online_hsv.get_sample_counts()
            print(
                f"[hsv_db] frame {fi} counts={counts} "
                f"ready={pipe._online_hsv.is_ready()} "
                f"injected={pipe._online_hsv_injected}",
            )
    cap.release()
    elapsed = time.time() - t0
    counts = pipe._online_hsv.get_sample_counts()
    ranges = pipe._online_hsv.get_per_video_ranges()
    state = pipe._online_hsv.export_state()
    payload = {
        "video_id": args.video_id,
        "video_path": str(args.video),
        "n_frames_processed": n_target,
        "elapsed_sec": float(elapsed),
        "sample_counts": counts,
        "per_video_ranges": {
            str(k): list(v) for k, v in ranges.items()
        },
        "online_hsv_state": state,
    }
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(
        f"[hsv_db] DONE saved {args.out}: counts={counts} "
        f"ranges_keys={list(ranges)} elapsed={elapsed:.1f}s",
    )


if __name__ == "__main__":
    main()
