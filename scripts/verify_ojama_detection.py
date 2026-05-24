"""
予告お邪魔検出の検証スクリプト

複数の試合中フレームに対して OjamaWarningDetector を実行し、
結果を 1 枚の検証画像 `data/verify/ojama_warning_check.png` にまとめる。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.ojama_warning import (
    P1_BOARD_X,
    P2_BOARD_X,
    WARNING_BOTTOM_Y,
    WARNING_TOP_Y,
    OjamaWarningDetector,
)

# 検証対象フレーム
TARGET_FRAMES: list[Path] = [
    Path("data/frames/sample/frame_2700s.png"),
    Path("data/frames/sample/frame_3200s.png"),
    Path("data/frames/sample/frame_0600s.png"),
    Path("data/frames/review_video_02/frame_0285s.png"),
    Path("data/frames/review_video_02/frame_0270s.png"),
]
OUTPUT_PATH: Path = Path("data/verify/ojama_warning_check.png")

# 切り出した strip 1 行のサイズ (ROI 帯のみ)
STRIP_HEIGHT: int = WARNING_BOTTOM_Y - WARNING_TOP_Y
STRIP_WIDTH: int = 384
LABEL_HEIGHT: int = 60
PADDING: int = 8


def _annotate(
    img: np.ndarray, text: str, color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """ラベル付きの画像を縦に 1 段増やす。"""
    h, w = img.shape[:2]
    panel = np.full((LABEL_HEIGHT, w, 3), 30, dtype=np.uint8)
    cv2.putText(
        panel, text, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA,
    )
    return np.vstack([panel, img])


def _build_row(
    frame: np.ndarray, frame_label: str,
    detector: OjamaWarningDetector,
) -> np.ndarray:
    """1 フレーム分の検証行 (ラベル + 1P strip + 2P strip) を返す。"""
    p1, p2 = detector.detect(frame)
    p1_strip = frame[
        WARNING_TOP_Y:WARNING_BOTTOM_Y,
        P1_BOARD_X:P1_BOARD_X + STRIP_WIDTH,
    ]
    p2_strip = frame[
        WARNING_TOP_Y:WARNING_BOTTOM_Y,
        P2_BOARD_X:P2_BOARD_X + STRIP_WIDTH,
    ]
    p1_text = (
        f"1P total={p1.total_count} icons={[i.icon_type for i in p1.icons]}"
    )
    p2_text = (
        f"2P total={p2.total_count} icons={[i.icon_type for i in p2.icons]}"
    )
    p1_panel = _annotate(p1_strip, p1_text, (0, 200, 255))
    p2_panel = _annotate(p2_strip, p2_text, (255, 200, 0))

    # 横に連結 + 左にフレーム名ラベル
    spacer = np.full(
        (p1_panel.shape[0], PADDING, 3), 30, dtype=np.uint8,
    )
    body = np.hstack([p1_panel, spacer, p2_panel])
    label_panel = np.full(
        (body.shape[0], 220, 3), 60, dtype=np.uint8,
    )
    cv2.putText(
        label_panel, frame_label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return np.hstack([label_panel, body])


def main() -> None:
    """検証フレーム群を 1 枚の画像にまとめて出力する。"""
    detector = OjamaWarningDetector()
    rows: list[np.ndarray] = []
    for fp in TARGET_FRAMES:
        if not fp.exists():
            print(f"[skip] {fp} 存在しない")
            continue
        frame = cv2.imread(str(fp))
        if frame is None or frame.shape[:2] != (1080, 1920):
            print(f"[skip] {fp} サイズ不一致")
            continue
        rows.append(_build_row(frame, fp.name, detector))
    if not rows:
        print("検証フレームがありません")
        return
    width = max(r.shape[1] for r in rows)
    padded: list[np.ndarray] = []
    for r in rows:
        pad_w = width - r.shape[1]
        if pad_w > 0:
            r = np.hstack([r, np.zeros((r.shape[0], pad_w, 3), dtype=np.uint8)])
        padded.append(r)
        padded.append(np.zeros((PADDING, width, 3), dtype=np.uint8))
    final = np.vstack(padded)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), final)
    print(f"出力: {OUTPUT_PATH}  shape={final.shape}")


if __name__ == "__main__":
    main()
