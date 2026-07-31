"""settle検出ロジック検証用の一時デバッグ: 特定イベントの diff 生時系列を出す。

_diag_chain_anim_duration_multi.py の SETTLE_MIN_SEC=0.5 が早期停止している
疑いを検証するため、代表的な chain_count=8 イベントの diff 時系列を frame毎に
ダンプする (read-only、src/ 変更なし)。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "2")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402

VIDEO_DIR = PROJ_ROOT / "data" / "frames"
OUT_DIR = PROJ_ROOT / "data" / "verify" / "_tmp_settle_trace_inspect"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (video_stem, fire_side, t_chain_start, chain_count) 代表サンプル
EVENTS = [
    ("c11", "1P", 451.400, 8),
    ("c16", "2P", 1118.400, 8),
    ("c21", "2P", 755.200, 5),
]
WINDOW_SEC = 20.0


def _region_gray(frame, region):
    x1, y1 = region.x, region.y
    x2, y2 = region.x + region.width, region.y + region.height
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)


def main() -> None:
    for stem, side, t0, cc in EVENTS:
        region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
        video_path = VIDEO_DIR / f"video_{stem}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        start_fi = int(round(t0 * fps))
        end_fi = int(round((t0 + WINDOW_SEC) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_fi))
        prev = None
        rows = []
        fi = start_fi
        while fi <= end_fi:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            gray = _region_gray(frame, region)
            diff = 0.0
            if prev is not None:
                diff = float(np.abs(gray.astype(np.int16) - prev.astype(np.int16)).mean())
            prev = gray
            rows.append((fi / fps - t0, diff))
            fi += 1
        cap.release()
        out_csv = OUT_DIR / f"{stem}_{side}_cc{cc}_t{t0:.0f}.csv"
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("t_rel,diff\n")
            for t_rel, diff in rows:
                f.write(f"{t_rel:.4f},{diff:.3f}\n")
        print(f"[done] {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
