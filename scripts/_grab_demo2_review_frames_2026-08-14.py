"""デモ2 (video_74・未見) の検収用フレーム抽出 (read-only, 2026-08-14).

data/verify/demo_fixed_2026-08-13/frames_demo2/ に保存する。
全域10秒間隔サンプリング + 試合境界の密サンプリング + ランダム3箇所の密チェック
(認識文字と実際の盤面の突合用)。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO = Path("data/verify/demo_fixed_2026-08-13/demo2_video74_3match.mp4")
OUT_DIR = Path("data/verify/demo_fixed_2026-08-13/frames_demo2")

# 全域10秒間隔 (0-180秒、全域無悪化/健全性チェック用)
GLOBAL_SECS = list(range(0, 181, 10))

# 試合境界候補の密サンプリング (source 284/340/406 - 230 = 54/110/176)
BOUNDARY_SECS = []
for b in (54, 110, 176):
    BOUNDARY_SECS.extend(range(b - 4, b + 5))

# ランダム3箇所の密チェック (認識文字と実際の盤面の突合、各箇所前後2秒 x 0.5秒刻み)
random.seed(20260814)
RANDOM_CENTERS = sorted(random.sample(range(15, 170), 3))
RANDOM_SECS: list[float] = []
for c in RANDOM_CENTERS:
    for off in (-1.0, -0.5, 0.0, 0.5, 1.0):
        RANDOM_SECS.append(round(c + off, 1))

print(f"[info] random dense centers: {RANDOM_CENTERS}")

ALL_SECS = sorted(set(GLOBAL_SECS) | set(BOUNDARY_SECS) | set(RANDOM_SECS))


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
        tag = "t"
        if t in RANDOM_SECS:
            tag = "rand_t"
        elif t in BOUNDARY_SECS:
            tag = "bnd_t"
        out_path = OUT_DIR / f"demo2_{tag}{t:06.1f}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"[saved] {out_path}")
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
