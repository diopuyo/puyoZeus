"""W: 1 動画分の (state_features, label) を npz として個別保存。

phase_w_batch_build_training.sh から並列実行されることを想定。
出力: data/training_phase_w/per_video/win_pred_v{NN}.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2
import numpy as np

from src.state_features import encode_state
from src.state_pipeline import StatePipeline


def load_winners(path: Path) -> list[dict]:
    import csv
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                rows.append({
                    "idx": int(r["idx"]),
                    "start_sec": float(r["start_sec"]),
                    "end_sec": float(r["end_sec"]),
                    "winner": r["winner"].strip(),
                })
            except (KeyError, ValueError):
                continue
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--interval", type=float, default=4.0)
    parser.add_argument(
        "--out-dir", default="data/training_phase_w/per_video",
    )
    parser.add_argument("--skip-seconds", type=float, default=5.0)
    args = parser.parse_args()

    vid_short = f"v{args.video_id:02d}"
    video_path = Path(f"data/frames/video_{args.video_id:02d}.mp4")
    winners_path = Path(f"data/verify/match_winners_{vid_short}.tsv")
    out_path = Path(args.out_dir) / f"win_pred_{vid_short}.npz"

    if not video_path.exists() or not winners_path.exists():
        print(f"missing: {video_path} or {winners_path}")
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 1
    pipeline = StatePipeline()
    matches = load_winners(winners_path)

    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    all_match_ids: list[int] = []

    for m in matches:
        winner = m["winner"]
        if winner not in ("1P", "2P"):
            continue
        label = 1 if winner == "1P" else 0
        pipeline.reset(match_start_sec=m["start_sec"])
        t = m["start_sec"] + args.skip_seconds
        end_t = m["end_sec"] - args.skip_seconds
        while t <= end_t:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            if not ok or fr is None:
                t += args.interval
                continue
            try:
                state = pipeline.extract(fr, t_sec=t)
                if not state.is_match_end_locked:
                    features = encode_state(state)
                    all_features.append(features)
                    all_labels.append(label)
                    all_match_ids.append(m["idx"])
            except Exception:
                pass
            t += args.interval

    cap.release()
    if not all_features:
        print(f"{vid_short}: no samples")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    X = np.stack(all_features).astype(np.float32)
    y = np.array(all_labels, dtype=np.int64)
    mids = np.array(all_match_ids, dtype=np.int64)
    video_ids = np.full(len(y), args.video_id, dtype=np.int64)
    np.savez_compressed(
        out_path, features=X, labels=y,
        match_ids=mids, video_ids=video_ids,
    )
    print(f"{vid_short}: {X.shape[0]} samples, "
          f"1P={int(y.sum())}/{len(y)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
