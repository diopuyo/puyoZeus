"""検収セルフベリファイ (demo2 v4) 補助: グラフ境界の目視用に境界直後/直前を追加抜き出す (read-only)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO = Path("data/verify/demo_fixed_2026-08-13/demo2_video74_3match.mp4")
OUT_DIR = Path("data/verify/demo_fixed_2026-08-13/frames_demo2_v4")


def grab(cap: cv2.VideoCapture, fps: float, t: float, name: str) -> None:
    frame_idx = int(round(t * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        print(f"[skip] t={t}s フレーム取得失敗")
        return
    cv2.imwrite(str(OUT_DIR / name), frame)
    print(f"[saved] {name} (t={t:.2f}s)")


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for t, name in [
        (56.0, "C3_graph_after_bnd1_t56.00.png"),
        (60.0, "C3_graph_after_bnd1_t60.00.png"),
        (108.0, "C3_graph_late_match2_t108.00.png"),
        (112.0, "C3b_graph_after_bnd2_t112.00.png"),
        (116.0, "C3b_graph_after_bnd2_t116.00.png"),
    ]:
        grab(cap, fps, t, name)
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
