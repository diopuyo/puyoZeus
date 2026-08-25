"""改修デモ (demo_fixed_3match.mp4) の検収用フレーム抽出 (read-only, 2026-08-13).

data/verify/demo_fixed_2026-08-13/frames/ に保存する。
台帳 docs/DEMO_REVIEW_2026-08-13.md の指摘場面 (#1/#2/#3/#8) を新デモの
相対時刻 (旧デモと同じ) で密抽出 + 全域10秒間隔サンプリング。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO = Path("data/verify/demo_fixed_2026-08-13/demo_fixed_3match.mp4")
OUT_DIR = Path("data/verify/demo_fixed_2026-08-13/frames")

# 全域10秒間隔 (0-148秒、全域無悪化チェック用)
GLOBAL_SECS = list(range(0, 148, 10))

# #1 (26秒付近): 2P OJAMA_FALL 張り付き確認
SCENE1_SECS = [20, 22, 24, 25, 26, 27, 28, 29, 30]

# #2 (34秒): 1P連鎖中の状態ラベル + 33%→反転の速さ
SCENE2_SECS = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

# #3 (73-83秒): 応手確率表示形式 + 判定の極端さ
SCENE3_SECS = [70, 72, 73, 75, 77, 79, 81, 83, 85, 87]

ALL_SECS = sorted(set(GLOBAL_SECS + SCENE1_SECS + SCENE2_SECS + SCENE3_SECS))


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"cannot open {VIDEO}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for t in ALL_SECS:
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[skip] t={t} frame_idx={frame_idx} 読み込み失敗", file=sys.stderr)
            continue
        out_path = OUT_DIR / f"fixed_t{t:04d}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"[saved] {out_path}")
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
