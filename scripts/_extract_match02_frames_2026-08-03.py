"""match_02 (olRyxDGacbg game_idx=2) 2P沈黙区間の実画面フレーム抽出 (2026-08-03)。

t_secはnpz記録時刻=元動画内の絶対秒 (render_delta_winprob_demo.py が
同じt_secをcv2.VideoCapture位置指定に使っている前提と同一)。main指示の
t=2996/3000/3004/3006 をフルスクリーンで抽出し、目視所見用に保存する。
"""
from __future__ import annotations

from pathlib import Path

import cv2

VIDEO_PATH = Path("data/frames/video_olRyxDGacbg.mp4")
OUT_DIR = Path("data/verify/match02_terminal_gap_frames_2026-08-03")
TARGET_TIMES_SEC = [2996.0, 3000.0, 3004.0, 3006.0]


def extract_frame_at(cap: cv2.VideoCapture, t_sec: float, fps: float) -> "cv2.Mat | None":
    frame_idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[info] fps={fps:.2f}")
    for t in TARGET_TIMES_SEC:
        frame = extract_frame_at(cap, t, fps)
        if frame is None:
            print(f"[skip] t={t:.1f}s フレーム取得失敗")
            continue
        out_path = OUT_DIR / f"t{t:.1f}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"[saved] {out_path}")
    cap.release()


if __name__ == "__main__":
    main()
