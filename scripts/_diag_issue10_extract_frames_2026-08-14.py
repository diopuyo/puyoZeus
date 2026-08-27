"""指摘#10 証拠フレーム抽出 (本体パイプライン非経由、生ソース動画からの直接切り出し)。

data/frames/review_demo_2026-08-12.mp4 の source t=194〜201 を1秒刻みで
PNG保存する (デモt=33-38 = source t=195-200 に前後1秒の余白を付けた範囲)。
"""
from __future__ import annotations

from pathlib import Path

import cv2

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT_DIR = Path("data/verify/demo_fixed_2026-08-13/frames_issue10")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMES_SEC = [194.0, 195.0, 196.0, 197.0, 198.0, 199.0, 200.0, 201.0]


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[info] fps={fps}")
    for t in TIMES_SEC:
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[skip] t={t}s frame_idx={frame_idx} 読み込み失敗")
            continue
        out_path = OUT_DIR / f"source_t{t:.0f}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"[saved] {out_path}")
    cap.release()


if __name__ == "__main__":
    main()
