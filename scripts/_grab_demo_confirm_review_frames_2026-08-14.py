"""確認デモ (集大成版, demo_fixed_3match.mp4, 18:13完成) の検収用フレーム抽出
(read-only, 2026-08-14)。

data/verify/demo_fixed_2026-08-13/frames_confirm/ に保存する。
指摘#10/#11の対象区間 (t=28-46秒) を密抽出 + #1/#2/#3/#8非退行確認点 +
全域10秒間隔サンプリング。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO = Path("data/verify/demo_fixed_2026-08-13/demo_fixed_3match.mp4")
OUT_DIR = Path("data/verify/demo_fixed_2026-08-13/frames_confirm")

# 全域10秒間隔 (0-148秒、全域無悪化チェック用)
GLOBAL_SECS = list(range(0, 148, 10))

# #10/#11 (t=28-46秒): 両者発火ホールド値+着弾完了までの延長確認 (メイン論点)
HOLD_SECS = list(range(28, 47))

# #1 (20-30秒付近): OJAMA_FALL 張り付き非退行確認
SCENE1_SECS = [20, 22, 24, 25, 26, 27, 28, 29, 30]

# #3 (70-88秒): 応手確率表示形式 + 判定の極端さの非退行確認
SCENE3_SECS = [70, 72, 73, 75, 77, 79, 81, 83, 85, 87]

# #8 (試合境界 約55秒/約111秒): グラフ相対時間リセットの非退行確認
SCENE8_SECS = [50, 53, 55, 56, 58, 60, 105, 108, 111, 112, 114, 116]

ALL_SECS = sorted(set(GLOBAL_SECS + HOLD_SECS + SCENE1_SECS + SCENE3_SECS + SCENE8_SECS))


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
        out_path = OUT_DIR / f"confirm_t{t:04d}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"[saved] {out_path}")
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
