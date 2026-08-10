"""スモーク検収用フレーム抽出 (2026-08-08 追記: t39.2周辺の復帰確認用)."""
from __future__ import annotations

import cv2

VIDEO = "data/verify/youtube_demo_2026-08-07/_smoke_cellstab.mp4"
OUT_DIR = "data/verify/youtube_demo_2026-08-07/_cellstab_frames"
TIMES = [38.5, 39.2, 39.7, 40.5, 41.0]


def main() -> None:
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    for t in TIMES:
        fi = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            print(f"read failed for t={t}")
            continue
        out_path = f"{OUT_DIR}/t_{t}.png"
        cv2.imwrite(out_path, frame)
        print(f"saved t={t} fps={fps} fi={fi} -> {out_path}")


if __name__ == "__main__":
    main()
