# 診断: 開始窓の不変非空セルを実画面フレームで検証する (2026-08-19)
# cv2.VideoCapture のみ使用 (bundled ffmpeg は本件動画で SIGSEGV のため禁止)。
# 各ケースについて 境界時刻/窓開始/窓終端 のフレームを保存し、該当セルに
# 矩形を描いた注釈つきフルフレームも保存する。
from __future__ import annotations

import os
import sys

import cv2

OUTDIR = "logs/diag_startwindow_frames_2026-08-19"

# 盤面ジオメトリ (src/image_reader.py DEFAULT_P1/P2_REGION 準拠、1920x1080)
REGION = {"1P": (282, 160), "2P": (1258, 160)}
CELL_W, CELL_H = 64, 60

# (video, mp4path, side, [(label, t_sec)], cells=[(row,col,color)])
CASES = [
    ("29_g14", "data/frames/video_29.mp4", "2P",
     [("boundary", 943.33), ("win_lo", 944.93), ("win_hi", 945.77)],
     [(9, 0, 3), (9, 1, 3)]),
    ("29_g24", "data/frames/video_29.mp4", "2P",
     [("boundary", 1346.73), ("win_lo", 1347.73), ("win_hi", 1349.47)],
     [(9, 0, 4)]),
    ("29_g54", "data/frames/video_29.mp4", "1P",
     [("boundary", 2627.63), ("win_lo", 2629.27), ("win_hi", 2630.07)],
     [(8, 2, 1), (9, 2, 1)]),
    ("38_g2", "data/frames/video_38.mp4", "1P",
     [("boundary", 229.83), ("win_lo", 231.67), ("win_hi", 232.53)],
     [(9, 2, 1)]),
    ("38_g16", "data/frames/video_38.mp4", "2P",
     [("boundary", 666.10), ("win_lo", 667.43), ("win_hi", 668.60)],
     [(9, 2, 5)]),
    ("39_g10_1P", "data/frames/video_39.mp4", "1P",
     [("boundary", 448.70), ("win_lo", 450.47), ("win_hi", 451.23)],
     [(9, 0, 4)]),
    ("39_g10_2P", "data/frames/video_39.mp4", "2P",
     [("boundary", 448.70), ("win_lo", 450.43), ("win_hi", 451.30)],
     [(9, 0, 4)]),
    ("c109_g28", "data/frames/video_c109.mp4", "2P",
     [("boundary", 1839.17), ("win_lo", 1840.50), ("win_hi", 1841.33)],
     [(9, 0, 3)]),
    ("c109_g170_1P", "data/frames/video_c109.mp4", "1P",
     [("boundary", 9133.00), ("win_lo", 9134.60), ("win_hi", 9135.50)],
     [(9, 0, 4), (9, 1, 4)]),
    ("c109_g187_1P", "data/frames/video_c109.mp4", "1P",
     [("boundary", 9957.13), ("win_lo", 9958.73), ("win_hi", 9959.50)],
     [(9, 0, 1)]),
]

# c100 (再DL後に有効化): game_idx=2 2P、残留 6セル (r9c0=5 ほか)
CASES_C100 = [
    ("c100_g2", "logs/diag_startwindow_frames_2026-08-19/video_c100_head.mp4", "2P",
     [("boundary", 340.03), ("t340.8", 340.77), ("win_lo", 341.10),
      ("win_mid", 342.50), ("win_hi", 343.30), ("t343.4", 343.43)],
     [(9, 0, 5), (10, 0, 1), (11, 0, 1), (11, 1, 1), (12, 0, 2), (12, 1, 2)]),
]


def annotate(frame, side: str, cells) -> None:
    x0, y0 = REGION[side]
    for r, c, col in cells:
        # grid row 0 = 隠し段 (画面外)。可視行は row1..12 → 表示 y は (r-1)。
        x = x0 + c * CELL_W
        y = y0 + (r - 1) * CELL_H
        cv2.rectangle(frame, (x, y), (x + CELL_W, y + CELL_H), (255, 255, 255), 2)
        cv2.putText(frame, f"r{r}c{c}={col}", (x - 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    cases = list(CASES)
    if len(sys.argv) > 1 and sys.argv[1] == "c100":
        cases = CASES_C100
    cap_cache: dict[str, cv2.VideoCapture] = {}
    for name, path, side, times, cells in cases:
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} なし")
            continue
        if path not in cap_cache:
            cap_cache[path] = cv2.VideoCapture(path)
        cap = cap_cache[path]
        fps = cap.get(cv2.CAP_PROP_FPS)
        for label, t in times:
            fi = int(round(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                print(f"[NG] {name} {label} t={t} frame read 失敗")
                continue
            if frame.shape[1] != 1920:
                frame = cv2.resize(frame, (1920, 1080))
            raw = os.path.join(OUTDIR, f"{name}_{label}_t{t:.2f}_raw.png")
            cv2.imwrite(raw, frame)
            ann = frame.copy()
            annotate(ann, side, cells)
            cv2.imwrite(os.path.join(OUTDIR, f"{name}_{label}_t{t:.2f}_ann.png"), ann)
            print(f"[OK] {name} {label} t={t:.2f} fps={fps:.2f} -> {raw}")
    for cap in cap_cache.values():
        cap.release()


if __name__ == "__main__":
    main()
