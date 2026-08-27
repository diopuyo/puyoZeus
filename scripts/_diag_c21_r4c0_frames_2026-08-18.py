"""004_c21 r4c0 (1P) の真値確認用: 生動画から直接フレームを切り出す (2026-08-18)。

計装のみ・パイプライン非経由・軽量 (単一動画・数フレームのみ)。
本体コードは変更しない。
"""
from __future__ import annotations

from pathlib import Path

import cv2

VIDEO_PATH = Path.home() / "frames" / "video_c21.mp4"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "verify" / "diag_c21_r4c0_2026-08-18"

# DEFAULT_P1_REGION: x=282, y=160, width=384, height=720 (12行×6列, 60px/行, 64px/列)
P1_X, P1_Y, P1_W, P1_H = 282, 160, 384, 720
CELL_H = P1_H / 12.0
CELL_W = P1_W / 6.0

# 調査対象フレーム (frame_idx, ラベル)
TARGETS = [
    (144300, "before_empty_long"),
    (144390, "before_empty2"),
    (144415, "ojama_fall_state_enter1"),
    (144434, "ojama_fall_state_exit1"),
    (144463, "ojama_fall_state_enter2"),
    (144470, "ojama_fall_mid2"),
    (144473, "cnn_first_flip_to_9"),
    (144480, "hist_building"),
    (144482, "confirmed_9"),
    (144486, "anchor_frame"),
    (144600, "later_still_9"),
    (145184, "chunk_end_9"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"video fps={fps}")
    for frame_idx, label in TARGETS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[warn] frame {frame_idx} 読み込み失敗")
            continue
        # 元動画が1920x1080以外の場合は1920x1080へリサイズ (CLAUDE.md規約、
        # RecognitionPipelineの実挙動と一致させる)
        h0, w0 = frame.shape[:2]
        if (w0, h0) != (1920, 1080):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        # 1P盤面ROI全体
        roi = frame[P1_Y : P1_Y + P1_H, P1_X : P1_X + P1_W]
        # 対象セル周辺 (行2〜7、列0〜2) を拡大切り出し (context込み)
        r0, r1 = 2, 8
        c0, c1 = 0, 3
        y0, y1 = int(r0 * CELL_H), int(r1 * CELL_H)
        x0, x1 = int(c0 * CELL_W), int(c1 * CELL_W)
        context = roi[y0:y1, x0:x1]
        # 4倍拡大
        context_big = cv2.resize(
            context, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST
        )
        # r4c0セルに赤枠 (画像行=r-1、行2起点なのでオフセット調整)
        target_r, target_c = 4, 0
        ry0 = int((target_r - 1 - r0) * CELL_H * 4)
        ry1 = int((target_r - r0) * CELL_H * 4)
        rx0 = int((target_c - c0) * CELL_W * 4)
        rx1 = int((target_c + 1 - c0) * CELL_W * 4)
        cv2.rectangle(context_big, (rx0, ry0), (rx1, ry1), (0, 0, 255), 2)
        out_path = OUT_DIR / f"f{frame_idx:06d}_{label}.png"
        cv2.imwrite(str(out_path), context_big)
        print(f"[ok] {out_path}")
    cap.release()


if __name__ == "__main__":
    main()
