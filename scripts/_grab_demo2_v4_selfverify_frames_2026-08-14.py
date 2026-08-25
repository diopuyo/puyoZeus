"""検収セルフベリファイ (demo2 v4): 完成済みmp4から実画面フレームを抜き出す (read-only)。

対象:
  A) 末尾5秒 (1秒刻み) - 4試合目の頭混入がないかの目視用
  B) ホールド区間 t=34-38秒 (0.5秒刻み) - #5修正 (settled凍結) の目視用
  C) 非退行スポット3点 (認識文字/応手表示/グラフ境界)

時刻は出力mp4のt=0基準 (デモとして視聴者が見る時間軸、start_sec=230の相対値)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO = Path("data/verify/demo_fixed_2026-08-13/demo2_video74_3match.mp4")
OUT_DIR = Path("data/verify/demo_fixed_2026-08-13/frames_demo2_v4")


def grab(cap: cv2.VideoCapture, fps: float, t: float, name: str) -> None:
    frame_idx = int(round(t * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        print(f"[skip] t={t}s フレーム取得失敗")
        return
    out_path = OUT_DIR / name
    cv2.imwrite(str(out_path), frame)
    print(f"[saved] {out_path} (t={t:.2f}s, frame_idx={frame_idx})")


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"cannot open {VIDEO}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps
    print(f"fps={fps} n_frames={n_frames} duration={duration:.3f}s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # A) 末尾5秒 (1秒刻み) + 最終フレーム
    tail_start = duration - 5.0
    for i in range(6):
        t = tail_start + i
        if t >= duration:
            t = duration - (1.0 / fps)
        grab(cap, fps, t, f"A_tail_t{t:06.2f}.png")
    grab(cap, fps, duration - (1.0 / fps), "A_tail_lastframe.png")

    # B) ホールド区間 t=34-38秒 (0.5秒刻み)
    t = 34.0
    while t <= 38.0 + 1e-6:
        grab(cap, fps, t, f"B_hold_t{t:05.2f}.png")
        t += 0.5

    # C) 非退行スポット3点
    grab(cap, fps, 10.0, "C1_recog_text_t10.00.png")  # 認識文字 (state表示)
    grab(cap, fps, 101.0, "C2_counter_display_t101.00.png")  # 応手表示
    grab(cap, fps, 55.0, "C3_graph_boundary_t55.00.png")  # グラフ境界 (1→2試合目付近)
    grab(cap, fps, 111.0, "C3b_graph_boundary_t111.00.png")  # グラフ境界 (2→3試合目付近)

    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
