"""
予告お邪魔アイコンの ROI 探索ユーティリティ。

実フレーム画像から 1P/2P 盤面上部の予告領域を切り出して
data/verify/ に保存する。目視で座標を確認した上で
src/ojama_warning.py の定数 (P1_TOP_LEFT_X など) を確定する。

使い方:
    python -m scripts.find_ojama_roi
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# 1920×1080 フレーム前提の盤面 ROI (calibration_video01.json と一致)
P1_BOARD_X: int = 282
P2_BOARD_X: int = 1258
BOARD_W: int = 384
BOARD_TOP_Y: int = 160

# 予告アイコン帯: 盤面の上 55px 程度
WARNING_TOP_Y: int = 105
WARNING_BOTTOM_Y: int = 160
WARNING_HEIGHT: int = WARNING_BOTTOM_Y - WARNING_TOP_Y

# 6 セルに分割した個別アイコンの中心 x オフセット
ICON_CELL_W: int = BOARD_W // 6  # = 64
ICON_HALF_W: int = 30            # 切り出し時の左右マージン

VERIFY_DIR: Path = Path("data/verify/ojama_roi")
SAMPLE_FRAMES: list[Path] = [
    Path("data/frames/sample/frame_2700s.png"),
    Path("data/frames/sample/frame_3200s.png"),
    Path("data/frames/sample/frame_2100s.png"),
    Path("data/frames/review_video_02/frame_0285s.png"),
    Path("data/frames/review_video_02/frame_0270s.png"),
]


def _crop_warning_strip(frame: np.ndarray, board_x: int) -> np.ndarray:
    """指定盤面の上方予告帯を 1 枚に切り出す。"""
    return frame[WARNING_TOP_Y:WARNING_BOTTOM_Y, board_x:board_x + BOARD_W]


def _crop_icon_cell(
    frame: np.ndarray, board_x: int, idx: int,
) -> np.ndarray:
    """6 セル中 idx 番目のアイコンを切り出す (中心±ICON_HALF_W)。"""
    cx = board_x + (idx + 0.5) * ICON_CELL_W
    cx_int = int(cx)
    return frame[
        WARNING_TOP_Y:WARNING_BOTTOM_Y,
        max(0, cx_int - ICON_HALF_W):cx_int + ICON_HALF_W,
    ]


def _process_frame(frame_path: Path, out_dir: Path) -> None:
    """1 フレームについて 1P/2P 帯と各アイコンを保存する。"""
    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"[skip] 読み込み失敗: {frame_path}")
        return
    if img.shape[:2] != (1080, 1920):
        print(f"[skip] サイズ不一致: {frame_path} {img.shape}")
        return
    name = frame_path.stem
    sub = out_dir / name
    sub.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sub / "p1_strip.png"), _crop_warning_strip(img, P1_BOARD_X))
    cv2.imwrite(str(sub / "p2_strip.png"), _crop_warning_strip(img, P2_BOARD_X))
    for i in range(6):
        cv2.imwrite(
            str(sub / f"p1_icon_{i}.png"),
            _crop_icon_cell(img, P1_BOARD_X, i),
        )
        cv2.imwrite(
            str(sub / f"p2_icon_{i}.png"),
            _crop_icon_cell(img, P2_BOARD_X, i),
        )
    print(f"[ok] {frame_path} -> {sub}")


def _print_confirmed_roi() -> None:
    """確定済み ROI 座標を出力する (src/ojama_warning.py に固定済みの値)。"""
    print("=" * 60)
    print("予告お邪魔 ROI 確定座標 (1920x1080 前提):")
    print(f"  ストリップ Y: {WARNING_TOP_Y}-{WARNING_BOTTOM_Y} "
          f"(高さ {WARNING_HEIGHT}px)")
    print(f"  1P 盤面 X: {P1_BOARD_X} 〜 {P1_BOARD_X + BOARD_W}")
    print(f"  2P 盤面 X: {P2_BOARD_X} 〜 {P2_BOARD_X + BOARD_W}")
    print(f"  6 セル等分時のセル幅: {ICON_CELL_W}px")
    print(f"  各アイコンの中心 X (1P): "
          f"{[P1_BOARD_X + int((i+0.5)*ICON_CELL_W) for i in range(6)]}")
    print(f"  各アイコンの中心 X (2P): "
          f"{[P2_BOARD_X + int((i+0.5)*ICON_CELL_W) for i in range(6)]}")
    print("=" * 60)


def main() -> None:
    """SAMPLE_FRAMES の全フレームについて ROI 切り出しを行う。"""
    _print_confirmed_roi()
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    for fp in SAMPLE_FRAMES:
        _process_frame(fp, VERIFY_DIR)


if __name__ == "__main__":
    main()
