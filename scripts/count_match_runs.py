"""
数値シグネチャの安定した run の数を数え、試合数を推定する。

手法:
    - 1 秒間隔で WIN パネル検出 + 両側数値シグネチャ
    - 連続する同一シグネチャを 1 run にまとめる（min_stable 秒以上）
    - run の transition 数 = 試合終了数

出力:
    stdout: run 一覧と試合数推定
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

from src.win_panel import WinPanelDetector


def _sig(patch: np.ndarray) -> np.ndarray:
    if patch is None or patch.size == 0:
        return np.zeros(256, dtype=np.uint8)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw.flatten().astype(np.uint8)


def _hamm(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--same-hamming", type=int, default=30,
                        help="同一シグネチャとみなすハミング距離上限")
    parser.add_argument("--min-stable-sec", type=float, default=3.0,
                        help="run として認める最低継続秒")
    args = parser.parse_args()

    detector = WinPanelDetector.load_default()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    # (t, panel, sigL, sigR) を列挙
    records: list[tuple[float, bool, np.ndarray, np.ndarray]] = []
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
        sigL = _sig(r.digit_left_roi) if r.present else np.zeros(256, dtype=np.uint8)
        sigR = _sig(r.digit_right_roi) if r.present else np.zeros(256, dtype=np.uint8)
        records.append((t, r.present, sigL, sigR))
        t += args.interval
    cap.release()

    # run 抽出: 連続するフレームの sig がほぼ同じなら同一 run
    runs: list[tuple[float, float, np.ndarray, np.ndarray]] = []  # (start, end, sigL, sigR)
    current_start: float | None = None
    current_L: np.ndarray | None = None
    current_R: np.ndarray | None = None
    current_end: float | None = None

    for t_sec, panel, sigL, sigR in records:
        if not panel:
            # run 終了
            if current_start is not None:
                assert current_L is not None and current_R is not None and current_end is not None
                if current_end - current_start >= args.min_stable_sec:
                    runs.append((current_start, current_end, current_L, current_R))
                current_start = current_end = None
                current_L = current_R = None
            continue
        if current_start is None:
            current_start = t_sec
            current_L, current_R = sigL, sigR
            current_end = t_sec
            continue
        assert current_L is not None and current_R is not None
        dL = _hamm(sigL, current_L)
        dR = _hamm(sigR, current_R)
        if dL <= args.same_hamming and dR <= args.same_hamming:
            # 継続
            current_end = t_sec
        else:
            # run 切れ
            if current_end is not None and current_end - current_start >= args.min_stable_sec:
                runs.append((current_start, current_end, current_L, current_R))
            current_start = t_sec
            current_L, current_R = sigL, sigR
            current_end = t_sec

    # 最後の run
    if current_start is not None and current_end is not None and current_end - current_start >= args.min_stable_sec:
        assert current_L is not None and current_R is not None
        runs.append((current_start, current_end, current_L, current_R))

    print(f"run 数: {len(runs)}  (= 試合数 + 1 の推定上限)")
    print(f"試合終了検出: {max(0, len(runs) - 1)}")
    print()
    print("run 一覧 (start, end, 継続秒):")
    for i, (s, e, _, _) in enumerate(runs):
        print(f"  #{i+1:3d}  t={s:6.1f}-{e:6.1f}  ({e-s:5.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
