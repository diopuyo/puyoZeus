"""W2.2: video × matches × winners から (state_features, 勝者ラベル) ペアを生成。

各試合の (start_sec, end_sec) 区間内を interval 秒ごとに StatePipeline で抽出し、
最終勝者を全フレームの label に付与する。複数動画 (v01/v02/v03) をまとめて
1 npz に保存する。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_build_training_data \
        --interval 2.0 --out data/training_phase_w/win_pred_train.npz
"""
from __future__ import annotations

import argparse
import csv
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

from src.state_features import TOTAL_FEATURE_DIM, encode_state
from src.state_pipeline import StatePipeline


def load_winners(path: Path) -> list[dict]:
    """match_winners_*.tsv をロード。"""
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


def extract_match_features(
    pipeline: StatePipeline,
    cap: cv2.VideoCapture,
    start_sec: float,
    end_sec: float,
    interval: float,
    skip_seconds: float = 5.0,
) -> list[np.ndarray]:
    """1 試合分の features を抽出。試合開始 skip_seconds は除外 (準備画面)。"""
    pipeline.reset(match_start_sec=start_sec)
    out: list[np.ndarray] = []
    t = start_sec + skip_seconds
    while t <= end_sec - skip_seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok or fr is None:
            t += interval
            continue
        try:
            state = pipeline.extract(fr, t_sec=t)
            # ロックダウン中・テロップで盤面 UNKNOWN 多すぎなどのフレームは除外
            if state.is_match_end_locked:
                t += interval
                continue
            features = encode_state(state)
            out.append(features)
        except Exception:
            pass
        t += interval
    return out


def process_video(
    video_path: Path,
    winners_path: Path,
    pipeline: StatePipeline,
    interval: float,
    max_matches: int = 0,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    """1 動画分の (features, labels, match_ids) を返す。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] cannot open: {video_path}")
        return [], [], []
    matches = load_winners(winners_path)
    if max_matches > 0:
        matches = matches[:max_matches]

    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    all_match_ids: list[int] = []

    for m in matches:
        winner = m["winner"]
        if winner not in ("1P", "2P"):
            continue
        label = 1 if winner == "1P" else 0
        feats = extract_match_features(
            pipeline, cap, m["start_sec"], m["end_sec"], interval,
        )
        n = len(feats)
        all_features.extend(feats)
        all_labels.extend([label] * n)
        all_match_ids.extend([m["idx"]] * n)
        print(
            f"  match {m['idx']}: {winner}, "
            f"{m['start_sec']:.0f}-{m['end_sec']:.0f}s, "
            f"{n} samples"
        )
    cap.release()
    return all_features, all_labels, all_match_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument(
        "--out", default="data/training_phase_w/win_pred_train.npz",
    )
    parser.add_argument(
        "--max-matches-per-video", type=int, default=0,
        help="動画ごとの最大試合数 (0=全部、デバッグ用)",
    )
    parser.add_argument(
        "--videos", nargs="+",
        default=["video_01", "video_02", "video_03"],
    )
    args = parser.parse_args()

    pipeline = StatePipeline()
    print("StatePipeline ready")

    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    all_match_ids: list[int] = []
    all_video_ids: list[int] = []

    for v_idx, vname in enumerate(args.videos, 1):
        video_path = Path(f"data/frames/{vname}.mp4")
        # video_01 -> v01
        vid_short = "v" + vname.split("_")[-1]
        winners_path = Path(f"data/verify/match_winners_{vid_short}.tsv")
        if not video_path.exists():
            print(f"[WARN] missing: {video_path}")
            continue
        if not winners_path.exists():
            print(f"[WARN] missing: {winners_path}")
            continue
        print(f"\n=== {vname} ===")
        feats, labels, match_ids = process_video(
            video_path, winners_path, pipeline,
            interval=args.interval,
            max_matches=args.max_matches_per_video,
        )
        all_features.extend(feats)
        all_labels.extend(labels)
        all_match_ids.extend(match_ids)
        all_video_ids.extend([v_idx] * len(feats))

    if not all_features:
        print("no samples generated")
        return 1

    X = np.stack(all_features)
    y = np.array(all_labels, dtype=np.int64)
    match_ids = np.array(all_match_ids, dtype=np.int64)
    video_ids = np.array(all_video_ids, dtype=np.int64)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=X.astype(np.float32),
        labels=y,
        match_ids=match_ids,
        video_ids=video_ids,
    )
    print(f"\nfinal: {X.shape}, label balance: 1P={int(y.sum())}/{len(y)}")
    print(f"saved: {to_windows_path(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
