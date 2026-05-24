"""中央テロップ「チャレンジャー リーグ 30先」のテンプレート画像を生成する。

m27 (1636-1713s) で試合終了告知として常時表示されるテロップが盤面中央右
寄りに被り、認識が破綻する。これを検出して盤面読取をロック (前盤面保持)
するためのテンプレートを作る。

入力:
    data/frames/video_01.mp4 の t=1670s フレーム
出力:
    models/ui_templates/telop_challenger.png  (中央〜右側のテロップ部分)
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2

VIDEO = "data/frames/video_01.mp4"
TIME_SEC = 1670.0
# 「チャレンジャー リーグ」ロゴ本体 (黄背景+黒文字) のみに絞る。
# 上部の「最大れんさ数」UI を含めると誤マッチするので除外。
# t=1670 の画像で確認: ロゴは x=750-1130, y=440-590
CROP_X = 770
CROP_Y = 440
CROP_W = 360
CROP_H = 150
OUT_PATH = Path("models/ui_templates/telop_challenger.png")


def main() -> int:
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"video open failed: {VIDEO}")
        return 1
    cap.set(cv2.CAP_PROP_POS_MSEC, TIME_SEC * 1000)
    ok, fr = cap.read()
    if not ok or fr is None:
        print("frame fetch failed")
        return 1
    if fr.shape[:2] != (1080, 1920):
        fr = cv2.resize(fr, (1920, 1080), interpolation=cv2.INTER_AREA)
    cap.release()

    crop = fr[CROP_Y:CROP_Y + CROP_H, CROP_X:CROP_X + CROP_W]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PATH), crop)
    print(f"[OK] {to_windows_path(str(OUT_PATH))}")
    print(f"     shape={crop.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
