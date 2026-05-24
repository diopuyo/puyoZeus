"""W3.2: 1 試合内での勝率推移を MLP で予測し、グラフ画像を生成。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_render_win_curve \
        --video data/frames/video_01.mp4 --start 186 --end 256 \
        --out data/verify/win_curve_v01_m1.png
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.state_features import encode_state
from src.state_pipeline import StatePipeline
from src.win_predictor import WinPredictorMLP


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--model", default="models/win_predictor_v2_mixed.pt",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--actual-winner", default=None,
        help="1P/2P を指定すると最終ラベルとしてグラフに表示",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"video open failed: {args.video}")
        return 1

    pipeline = StatePipeline()
    pipeline.reset(match_start_sec=args.start)
    model = WinPredictorMLP()
    model.load(args.model)

    times: list[float] = []
    probs: list[float] = []
    t = args.start
    while t <= args.end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok or fr is None:
            t += args.interval
            continue
        try:
            state = pipeline.extract(fr, t_sec=t)
            features = encode_state(state)
            prob = model.predict(features)
            times.append(t)
            probs.append(prob)
        except Exception as e:
            print(f"  err at t={t}: {e}")
        t += args.interval
    cap.release()

    if not times:
        print("no samples")
        return 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, probs, label="P(1P win)", color="#2080d0", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50/50")
    if args.actual_winner == "1P":
        ax.axhline(1.0, color="green", linestyle=":", alpha=0.7,
                   label="actual: 1P")
    elif args.actual_winner == "2P":
        ax.axhline(0.0, color="red", linestyle=":", alpha=0.7,
                   label="actual: 2P")
    ax.set_xlim(args.start, args.end)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("P(1P win)")
    ax.set_title(
        f"{Path(args.video).stem} ({args.start:.0f}-{args.end:.0f}s)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved: {to_windows_path(out_path)}")
    print(f"  samples: {len(times)}, mean p: {np.mean(probs):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
