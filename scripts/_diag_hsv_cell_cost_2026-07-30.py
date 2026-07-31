"""セル単位 HSV 分類の実測コスト内訳。

ステップ2 (セル単位 HSV のベクトル化) に工数を投じる前に、
**削減見込み 20ms が実測に耐えるか**を確かめる。
見積もりを実測と混同しない規律 (memory feedback_label_measured_vs_estimated_2026-07-30)。

計測対象 (ImageReader を monkeypatch して累積時間を取る):
  - classify()                : セル 1 個の HSV 分類全体
  - _compute_stable_h_median  : H median (赤色相折り返し補正込み)
  - _compute_specular_robust_s: S median (光沢除外込み)
  - cv2.cvtColor              : BGR→HSV 変換

出力は 1 フレームあたりの ms と、認識全体に対する比率。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_hsv_cell_cost_2026-07-30 \
        --video data/baseline_videos_v3/v29m2_buf15s.mp4 --frames 60
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

# 累積時間 [秒] と呼び出し回数
_ACC: dict[str, float] = defaultdict(float)
_CNT: dict[str, int] = defaultdict(int)


def _wrap(owner: Any, name: str, key: str) -> None:
    """メソッドを時間計測ラッパーで置き換える。"""
    original: Callable = getattr(owner, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _ACC[key] += time.perf_counter() - t0
            _CNT[key] += 1

    setattr(owner, name, wrapper)


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
        if frame.shape[1] != 1920 or frame.shape[0] != 1080:
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        out.append(frame)
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--start-sec", type=float, default=20.0)
    args = ap.parse_args()

    cv2.setNumThreads(1)

    from src.image_reader import ColorClassifier
    from src.recognition_pipeline import RecognitionPipeline

    _wrap(ColorClassifier, "classify", "classify")
    _wrap(ColorClassifier, "_compute_stable_h_median", "h_median")
    _wrap(ColorClassifier, "_compute_specular_robust_s", "s_median")

    frames = _read_frames(args.video, args.frames, args.start_sec)
    n = len(frames)
    print(f"フレーム数: {n} ({args.video.name} t={args.start_sec}s)")

    pipe = RecognitionPipeline.load_default()
    t0 = time.perf_counter()
    for idx, frame in enumerate(frames):
        pipe.update(idx, idx / 30.0, frame)
    total = time.perf_counter() - t0

    print(f"\n認識全体: {total:.2f}s = {total / n * 1000:.1f} ms/frame ({n / total:.2f} fps)")
    print(f"{'項目':<16}{'ms/frame':>10}{'全体比':>9}{'回/frame':>10}")
    for key in ("classify", "h_median", "s_median"):
        if _CNT[key] == 0:
            print(f"{key:<16}{'(呼出なし)':>10}")
            continue
        ms = _ACC[key] / n * 1000
        print(f"{key:<16}{ms:>10.1f}{100 * _ACC[key] / total:>8.1f}%{_CNT[key] / n:>10.1f}")

    # h_median + s_median が classify のどれだけを占めるか (= ベクトル化の上限効果)
    if _ACC["classify"] > 0:
        inner = _ACC["h_median"] + _ACC["s_median"]
        print(
            f"\nベクトル化の上限効果 (h+s median が classify に占める割合): "
            f"{100 * inner / _ACC['classify']:.1f}% "
            f"= {inner / n * 1000:.1f} ms/frame"
        )
        print(
            "※ classify 自体の残り (cvtColor・閾値ループ・Python 呼出) も "
            "ベクトル化すれば削減対象になるので、上限は classify 全体 "
            f"{_ACC['classify'] / n * 1000:.1f} ms/frame"
        )


if __name__ == "__main__":
    main()
