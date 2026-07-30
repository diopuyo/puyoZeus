"""1フレームあたり何セルが変化しているかを測る。

狙い: 収集の高速化。認識コストの大半はセル単位の分類(matchTemplate/CNN)なので、
「変化していないセルは前回の判定を流用する」ことで大幅に削れる可能性がある。
フレーム単位のゲートは既に否定済み(ぷよのアニメで盤面はほぼ常に変化する)。
しかし**変化しているのが78セルのうち数個だけ**なら、セル単位のゲートは有効。

差分計算そのものはピクセルの減算なので、matchTemplate に比べて桁違いに安い。
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.ojama_warning import BOARD_WIDTH, P1_BOARD_X, P2_BOARD_X

BOARD_TOP_Y: int = 180
CELL_PX: int = 64
N_ROWS: int = 13
N_COLS: int = 6
N_CELLS: int = N_ROWS * N_COLS
BOARD_BOTTOM_Y: int = BOARD_TOP_Y + CELL_PX * N_ROWS
TARGET_W, TARGET_H = 1920, 1080
# セルが「変化した」と見なす平均絶対差。ぷよ1個の出現はそのセルで約100の差になる
CELL_THRESHOLDS: tuple[float, ...] = (5.0, 10.0, 20.0)


def per_cell_diff(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """セルごとの平均絶対差 (N_ROWS, N_COLS) を返す。"""
    d = cv2.absdiff(cur, prev)
    return d.reshape(N_ROWS, CELL_PX, N_COLS, CELL_PX).mean(axis=(1, 3))


def main() -> None:
    """指定区間で変化セル数の分布を集計する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-sec", type=float, default=1800.0)
    ap.add_argument("--dur-sec", type=float, default=90.0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_sec * fps))
    n_frames = int(args.dur_sec * fps)
    print(f"fps={fps:.1f} 対象={n_frames}フレーム 盤面={N_CELLS}セル/side")

    prev: list[np.ndarray] | None = None
    counts: dict[float, list[int]] = {th: [] for th in CELL_THRESHOLDS}
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H),
                               interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur = [g[BOARD_TOP_Y:BOARD_BOTTOM_Y, x:x + BOARD_WIDTH]
               for x in (P1_BOARD_X, P2_BOARD_X)]
        if prev is not None:
            for th in CELL_THRESHOLDS:
                n = sum(int((per_cell_diff(cur[k], prev[k]) >= th).sum())
                        for k in (0, 1))
                counts[th].append(n)
        prev = cur
    cap.release()

    total = N_CELLS * 2  # 1P+2P
    print(f"\n両サイド合計 {total} セル中、1フレームで変化したセル数:")
    for th in CELL_THRESHOLDS:
        a = np.array(counts[th])
        if len(a) == 0:
            continue
        print(f"  閾値{th:>4.1f}: 中央値{np.median(a):5.1f} "
              f"平均{a.mean():5.1f} p90={np.percentile(a, 90):5.1f} "
              f"最大{a.max():3d} "
              f"→ **分類が必要なセルは平均{a.mean()/total*100:4.1f}%** "
              f"(削減率 {100-a.mean()/total*100:4.1f}%)")
        print(f"           変化0セルのフレーム: {(a == 0).mean()*100:4.1f}%")


if __name__ == "__main__":
    main()
