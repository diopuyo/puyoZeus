"""測定2 代表動画選定の目視確認用フレーム抽出 (2026-08-25)。

video_51.mp4 (マスター・3ブロック light vs SAKI、tier確認済み) のラウンド境界
候補 (t=459.00 開始 / t=533.50 終了、粗いスコアリセット走査で検出) について、
実画面フレームを PNG 保存し、後で Read ツールで目視確認する。

**本番コードは使わない** (cv2 のみ)。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

VIDEO = PROJECT_ROOT / "data/frames/video_51.mp4"
OUT_DIR = PROJECT_ROOT / "data/verify/gate3_episode_repr_2026-08-25/frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 目視確認したい時刻 (秒)。境界前後 + ラウンド中盤。
TIMES = [456.0, 458.0, 459.0, 460.0, 462.0, 495.0, 531.0, 533.0, 533.5, 534.0, 536.0]


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    for t in TIMES:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[warn] read failed at t={t}")
            continue
        out_path = OUT_DIR / f"t_{t:07.2f}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"[saved] t={t:.2f} -> {out_path}")
    cap.release()


if __name__ == "__main__":
    main()
