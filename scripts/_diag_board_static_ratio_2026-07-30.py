"""盤面領域が前フレームと実質同一なフレームの割合を測る。

狙い: 「安い変化検出で高い認識(matchTemplate)をゲートする」案が成立するかを事前に判定する。
静止フレームの割合が高ければゲートで削れる上限が大きい。低ければ実装する意味がない。
認識は一切走らせず、盤面ROIのピクセル差分のみを見る (数分で終わる)。

背景ステージは常にアニメーションしているため、必ず盤面ROIに限定して測ること。
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.ojama_warning import BOARD_WIDTH, P1_BOARD_X, P2_BOARD_X

# 盤面の縦範囲 (1920x1080前提、13行 x セル64px = 832px)
BOARD_TOP_Y: int = 180
BOARD_BOTTOM_Y: int = BOARD_TOP_Y + 64 * 13
TARGET_W, TARGET_H = 1920, 1080
# 「実質同一」と見なす閾値 (0-255スケール)。複数試して感度を見る
THRESHOLDS: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)
# セル単位の最大差の閾値。ぷよ1個の出現はそのセルで約100の差になるため、
# 雑音(数レベル)と桁で分離できる。盤面平均では1.28にしか見えず鈍い。
CELL_THRESHOLDS: tuple[float, ...] = (3.0, 5.0, 10.0, 20.0, 40.0)
CELL_PX: int = 64
N_ROWS: int = 13
N_COLS: int = 6


def board_rois(frame: np.ndarray) -> list[np.ndarray]:
    """1P/2P の盤面領域をグレースケールで返す。"""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = []
    for x0 in (P1_BOARD_X, P2_BOARD_X):
        out.append(g[BOARD_TOP_Y:BOARD_BOTTOM_Y, x0:x0 + BOARD_WIDTH])
    return out


def main() -> None:
    """指定区間の盤面差分を集計する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-sec", type=float, default=1800.0)
    ap.add_argument("--dur-sec", type=float, default=90.0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_sec * fps))
    n_frames = int(args.dur_sec * fps)
    print(f"fps={fps:.1f} 対象={n_frames}フレーム ({args.dur_sec:.0f}秒)")

    prev: list[np.ndarray] | None = None
    diffs: list[list[float]] = [[], []]
    cellmax: list[list[float]] = [[], []]
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H),
                               interpolation=cv2.INTER_AREA)
        cur = board_rois(frame)
        if prev is not None:
            for k in (0, 1):
                d = cv2.absdiff(cur[k], prev[k])
                diffs[k].append(float(d.mean()))
                # セル単位に集約して最大を取る (ぷよ1個の出現を確実に捉える)
                per_cell = d.reshape(N_ROWS, CELL_PX, N_COLS, CELL_PX).mean(
                    axis=(1, 3))
                cellmax[k].append(float(per_cell.max()))
        prev = cur
    cap.release()

    for k, name in ((0, "1P"), (1, "2P")):
        a = np.array(diffs[k])
        if len(a) == 0:
            continue
        print(f"\n=== {name} 盤面 (n={len(a)}) ===")
        print(f"  平均絶対差: 中央値{np.median(a):.2f} "
              f"p10={np.percentile(a, 10):.2f} p90={np.percentile(a, 90):.2f} "
              f"最大{a.max():.1f}")
        print("  [盤面平均で判定 = 鈍い]")
        for th in THRESHOLDS:
            r = float((a < th).mean())
            print(f"    閾値{th:>4.1f}未満で静止: {r*100:5.1f}% "
                  f"→ 削減率 {r*100:4.0f}%")
        c = np.array(cellmax[k])
        print(f"  [セル単位の最大差で判定 = 鋭い] "
              f"中央値{np.median(c):.1f} p90={np.percentile(c, 90):.1f} "
              f"最大{c.max():.0f}")
        for th in CELL_THRESHOLDS:
            r = float((c < th).mean())
            print(f"    閾値{th:>4.1f}未満で静止: {r*100:5.1f}% "
                  f"→ 削減率 {r*100:4.0f}%")


if __name__ == "__main__":
    main()
