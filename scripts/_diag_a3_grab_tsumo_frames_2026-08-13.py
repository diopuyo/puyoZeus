"""A-3 副次確認 (tsumo_count 2P>1P傾向) 用の実画面フレーム抽出 (read-only)。

認識パイプラインを一切実行せず cv2 のみでフレームを切り出す軽量処理
(9並列走行中の収集ジョブと競合しないよう CPU 負荷を最小化する)。
data/verify/a3_ojama_symmetry_2026-08-13/frames/ に保存する
(レビュー規約 feedback_review_actual_screen_frames_2026-07-24 準拠)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT_DIR = Path("data/verify/a3_ojama_symmetry_2026-08-13/frames")
# 場面3 (docs/DEMO_REVIEW_2026-08-13.md 「4試合目序盤・設置確定遅延」)
# t=311-335 の実プレイ帯を 2 秒間隔で抽出し、1P/2P 双方の手数進行を目視確認する。
TARGET_SECS = [300.0, 305.0, 310.0, 313.0, 316.0, 319.0, 322.0, 325.0,
               328.0, 331.0, 334.0, 337.0, 340.0]


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"cannot open {VIDEO}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for t in TARGET_SECS:
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[skip] t={t} frame_idx={frame_idx} 読み込み失敗", file=sys.stderr)
            continue
        out_path = OUT_DIR / f"t{t:07.2f}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"[saved] {out_path}")
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
