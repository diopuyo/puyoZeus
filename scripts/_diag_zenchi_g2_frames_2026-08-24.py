"""納品動画 seg01 (video_zenchi_c0BQoMJwwQU) game2 の t=173〜224 実画面フレーム切り出し。

user指摘「99%→1%急降下 (1本目の3分付近)」の実態確認用 (read-only、動画は無変更)。
出力: logs/zenchi_g2_frames_2026-08-24/t<秒>.png (960x540 縮小)。
"""
from __future__ import annotations

from pathlib import Path

import cv2

VIDEO = Path("data/frames/video_zenchi_c0BQoMJwwQU.mp4")
OUT_DIR = Path("logs/zenchi_g2_frames_2026-08-24")

# 切り出し時刻 (dump の t_sec と同じ基準 = フレーム番号/fps)
TIMES: list[float] = [
    173.0, 175.0, 176.5, 177.5, 179.0, 181.0, 183.0, 185.0,
    186.0, 186.7, 187.6, 189.0, 191.0, 194.0, 197.0, 199.0,
    200.5, 201.0, 201.5, 202.5, 204.0, 205.5, 206.5, 208.0,
    210.0, 212.0, 214.0, 216.0, 218.0, 220.0, 222.0, 223.0, 224.0,
]


def main() -> None:
    """指定時刻のフレームを縮小 PNG で保存する。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"fps={fps:.3f}")
    for t in TIMES:
        fi = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            print(f"t={t}: read失敗")
            continue
        small = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        p = OUT_DIR / f"t{t:07.2f}.png"
        cv2.imwrite(str(p), small)
        print(f"saved {p}")
    cap.release()


if __name__ == "__main__":
    main()
