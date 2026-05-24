"""
動画を走査して勝敗数値の変化点を見つけ、各数値のパッチ画像を保存する。

処理:
    - WinPanelDetector でパネル検出
    - パネルある場所で数値 ROI (左右) を保存
    - 色ヒストグラム で「同じ数値かどうか」判定し、変化点のみ保存
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

import cv2
import numpy as np

from src.win_panel import (
    WinPanelDetector,
    NUMBER_LEFT_X,
    NUMBER_RIGHT_X,
    NUMBER_Y,
)


def _digit_hash(patch: np.ndarray) -> str:
    """数値パッチのシンプルな指紋（同じ数値なら同じハッシュ）。"""
    if patch is None or patch.size == 0:
        return ""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    thr = (small > small.mean()).astype(np.uint8).flatten()
    return "".join(str(x) for x in thr.tolist())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--out", default="data/verify/digit_samples")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = WinPanelDetector.load_default()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    seen_hashes: set[str] = set()
    saved = 0
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            t += args.interval
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        r = detector.detect(frame)
        if not r.present:
            t += args.interval
            continue
        for label, roi in (("L", r.digit_left_roi), ("R", r.digit_right_roi)):
            if roi is None:
                continue
            h = _digit_hash(roi)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            out = out_dir / f"t{int(t):05d}_{label}.png"
            cv2.imwrite(str(out), roi)
            saved += 1
        t += args.interval
    cap.release()

    print(f"保存: {saved} 種類（ユニーク hash: {len(seen_hashes)}）")
    print(f"出力: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
