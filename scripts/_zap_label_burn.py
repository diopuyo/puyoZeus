"""ザッピングレビュー動画用: 各区間クリップの左上に動画ID/ティア/対戦カードを焼き込む。

visualize_advantage_overlay.py の出力(認識色 overlay 済み mp4v)を読み込み、
左上の空き領域(タイトル文字列より左、x<270)に3行ラベルを半透明ボックスで
焼いて再書き出しする。認識・推論は一切行わない(単純なフレームコピー+描画の
みのため高速)。src/ は無改修。2026-07-23 ザッピングレビュー動画タスク専用。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ラベルボックスのレイアウト定数 (マジックナンバー禁止 → 定数化)
LABEL_X: int = 8
LABEL_Y0: int = 8
LABEL_LINE_H: int = 26
LABEL_BOX_W: int = 260
LABEL_BOX_PAD: int = 6
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\meiryo.ttc", "/mnt/c/Windows/Fonts/meiryo.ttc",
)


def _font(size: int) -> ImageFont.ImageFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _draw_label(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    """1 フレームに3行ラベルを焼く (RGBA半透明ボックス + 文字)。"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    box_h = LABEL_LINE_H * len(lines) + LABEL_BOX_PAD * 2
    d.rectangle(
        [LABEL_X - LABEL_BOX_PAD, LABEL_Y0 - LABEL_BOX_PAD,
         LABEL_X + LABEL_BOX_W, LABEL_Y0 + box_h - LABEL_BOX_PAD],
        fill=(0, 0, 0, 160),
    )
    for i, (text, color) in enumerate(lines):
        d.text((LABEL_X, LABEL_Y0 + i * LABEL_LINE_H), text,
               font=_font(18), fill=color + (255,))
    img = Image.alpha_composite(img, overlay).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def burn(src: Path, dst: Path, video_id: str, tier: str, matchup: str) -> int:
    """src の全フレームにラベルを焼いて dst (mp4v) へ書き出す。書き出しフレーム数を返す。"""
    lines = [
        (video_id, (255, 255, 0)),
        (tier, (255, 180, 100)),
        (matchup, (255, 255, 255)),
    ]
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[ERROR] open失敗: {src}", file=sys.stderr)
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(_draw_label(frame, lines))
        n += 1
    cap.release()
    writer.release()
    print(f"[done] {n} frames -> {dst}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--matchup", required=True)
    a = ap.parse_args()
    burn(Path(a.src), Path(a.dst), a.video_id, a.tier, a.matchup)


if __name__ == "__main__":
    main()
