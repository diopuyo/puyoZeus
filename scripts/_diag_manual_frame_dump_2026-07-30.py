"""c60_g2 / c75_g0 のuser指定ウィンドウの実フレームを目視用に抽出する (2026-07-30 一時)。

検証2の3分岐判定のため、検出器が疑惑フラグを立てた事象(c75 1P t=262.0/264.2)を
中心に、userが目視した試合区間全体をカバーする間隔サンプルも合わせて保存する。
測定専用、修正は行わない。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.chain_count_ocr import _ensure_1080p  # noqa: E402
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402

OUT = PROJ_ROOT / "data" / "verify" / "frozen_verify_c60c75_2026-07-30" / "manual_check"
VIDEO_DIR = PROJ_ROOT / "data" / "frames"


def grab(video_stem: str, t: float):
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"video_{video_stem}.mp4"))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return _ensure_1080p(frame)


def save_full_and_boards(video_stem: str, t: float, tag: str) -> None:
    frame = grab(video_stem, t)
    if frame is None:
        print(f"[WARN] フレーム取得失敗 {video_stem} t={t}")
        return
    cv2.imwrite(str(OUT / f"{video_stem}_{tag}_full.png"), frame)
    for side, region in (("1P", DEFAULT_P1_REGION), ("2P", DEFAULT_P2_REGION)):
        crop = frame[region.y:region.y + region.height, region.x:region.x + region.width]
        h, w = crop.shape[:2]
        up = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT / f"{video_stem}_{tag}_{side}_board.png"), up)
    print(f"[OK] {video_stem} t={t:.1f} -> {tag}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # c60_g2: レンダジョブ指定314.0-400.0秒 (境界は2秒早い可能性考慮し312-402で見る)
    for t in (312.0, 350.0, 384.2, 390.0, 396.8, 400.0, 402.0):
        save_full_and_boards("c60", t, f"t{t:.1f}")
    # c75_g0: レンダジョブ指定200.0-268.0秒、かつ検出器が疑惑フラグを立てたt=262.0/264.2周辺
    for t in (198.0, 205.4, 233.6, 250.6, 260.0, 262.0, 263.0, 264.2, 265.0, 266.0, 268.0, 270.0):
        save_full_and_boards("c75", t, f"t{t:.1f}")


if __name__ == "__main__":
    main()
