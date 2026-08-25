"""W13根治 案1 効果測定用の実画面フレーム証拠切り出し (計装専用)。

data/verify/diag_issue16_2026-08-15/raw_frames/ (t=282〜292、既存資産) から
1P col0・2P col2 の列全体を切り出し、目視確認用に保存する。
"""
from __future__ import annotations

from pathlib import Path

import cv2

from src.board import BOARD_COLS, VISIBLE_ROWS
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION

SRC_DIR = Path("data/verify/diag_issue16_2026-08-15/raw_frames")
OUT_DIR = Path("data/verify/diag_w13_fix_2026-08-16/frames")

TARGET_TIMES = [282.00, 285.00, 288.00, 291.00]


def crop_column(frame, region, col: int):
    """region 全体のうち指定列 (0-5) のみを切り出す (可視12行分)。"""
    cw = region.width / BOARD_COLS
    x1 = int(region.x + col * cw)
    x2 = int(region.x + (col + 1) * cw)
    y1 = region.y
    y2 = region.y + region.height
    return frame[y1:y2, x1:x2]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for t in TARGET_TIMES:
        src = SRC_DIR / f"frame_t{t:.2f}_idx{int(t * 60)}.png"
        if not src.exists():
            print(f"skip (not found): {src}")
            continue
        frame = cv2.imread(str(src))
        if frame is None:
            print(f"skip (read fail): {src}")
            continue
        # 認識パイプラインは常に 1920x1080 に正規化してから読む
        # (CLAUDE.md 原則)。raw_frames は動画ネイティブ解像度 (720p) で
        # 保存されているため、同じ正規化をしてから座標系を合わせる。
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        # 1P 全体 + col0 切り出し
        p1_full = frame[
            DEFAULT_P1_REGION.y:DEFAULT_P1_REGION.y + DEFAULT_P1_REGION.height,
            DEFAULT_P1_REGION.x:DEFAULT_P1_REGION.x + DEFAULT_P1_REGION.width,
        ]
        p1_col0 = crop_column(frame, DEFAULT_P1_REGION, 0)
        p2_full = frame[
            DEFAULT_P2_REGION.y:DEFAULT_P2_REGION.y + DEFAULT_P2_REGION.height,
            DEFAULT_P2_REGION.x:DEFAULT_P2_REGION.x + DEFAULT_P2_REGION.width,
        ]
        p2_col2 = crop_column(frame, DEFAULT_P2_REGION, 2)
        cv2.imwrite(str(OUT_DIR / f"t{t:.2f}_1p_board.png"), p1_full)
        cv2.imwrite(str(OUT_DIR / f"t{t:.2f}_1p_col0.png"), p1_col0)
        cv2.imwrite(str(OUT_DIR / f"t{t:.2f}_2p_board.png"), p2_full)
        cv2.imwrite(str(OUT_DIR / f"t{t:.2f}_2p_col2.png"), p2_col2)
        print(f"saved: t={t}")


if __name__ == "__main__":
    main()
