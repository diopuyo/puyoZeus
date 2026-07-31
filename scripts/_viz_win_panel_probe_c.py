"""c 系 4 動画 (c1/c4/c34/c82) の WIN★パネル判定モーメントを画像化する検証専用 viz。

extract_match_winners.py の出力 JSON (winners_probe_2026-07-23/) を読み、各試合の
判定に使った 2 時点 (試合開始 offset 後 / 次試合開始 offset 後) のパネル領域を
実際にフレームから切り出して並べる。v 系 (v29 等) と同じパネル UI か目視確認する用途。

読み取り専用 viz。既存データは一切変更しない。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.win_panel import PANEL_X_RANGE, PANEL_Y_RANGE

WINNERS_DIR = Path("data/verify/winners_probe_2026-07-23")
FRAMES_DIR = Path("data/frames")
OUT_DIR = Path("data/verify")

TARGET_VIDEOS: list[str] = ["c1", "c4", "c34", "c82"]

WINNER_OFFSET_SEC: float = 2.0  # extract_match_winners.py と同じ値

# クロップ画像の拡大率・レイアウト定数
CROP_SCALE: int = 2
LABEL_HEIGHT: int = 24
ROW_GAP: int = 6
COL_GAP: int = 20
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _read_frame(cap: cv2.VideoCapture, t_sec: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def _crop_panel(frame: np.ndarray) -> np.ndarray:
    y1, y2 = PANEL_Y_RANGE
    x1, x2 = PANEL_X_RANGE
    roi = frame[y1:y2, x1:x2].copy()
    h, w = roi.shape[:2]
    return cv2.resize(roi, (w * CROP_SCALE, h * CROP_SCALE), interpolation=cv2.INTER_NEAREST)


def _label_row(text: str, width: int) -> np.ndarray:
    """指定幅のラベル帯 (黒背景 + 白文字) を返す。"""
    strip = np.zeros((LABEL_HEIGHT, width, 3), dtype=np.uint8)
    cv2.putText(strip, text, (4, LABEL_HEIGHT - 6), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return strip


def build_video_mosaic(video_id: str) -> np.ndarray | None:
    """1 動画分の全試合について before/after クロップを縦に並べたモザイクを返す。"""
    json_path = WINNERS_DIR / f"video_{video_id}.json"
    if not json_path.exists():
        print(f"  [WARN] winners JSON なし: {json_path}")
        return None
    with json_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    games = data.get("games", [])
    if not games:
        print(f"  [WARN] {video_id}: 試合記録なし")
        return None

    video_path = FRAMES_DIR / f"video_{video_id}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] 動画を開けない: {video_path}")
        return None

    rows: list[np.ndarray] = []
    for g in games:
        t_before = float(g["start_sec"]) + WINNER_OFFSET_SEC
        t_after = float(g["end_sec"]) + WINNER_OFFSET_SEC
        frame_before = _read_frame(cap, t_before)
        frame_after = _read_frame(cap, t_after)
        crop_before = _crop_panel(frame_before) if frame_before is not None else None
        crop_after = _crop_panel(frame_after) if frame_after is not None else None
        if crop_before is None and crop_after is None:
            continue

        h_ref = (crop_before if crop_before is not None else crop_after).shape[0]
        w_ref = (crop_before if crop_before is not None else crop_after).shape[1]
        blank = np.zeros((h_ref, w_ref, 3), dtype=np.uint8)
        if crop_before is None:
            crop_before = blank
        if crop_after is None:
            crop_after = blank

        gap = np.zeros((h_ref, COL_GAP, 3), dtype=np.uint8)
        row_img = np.concatenate([crop_before, gap, crop_after], axis=1)

        label = (
            f"game={g['game_abs_idx']:>2}  winner={str(g['winner']):<4}  "
            f"conf={g['confidence']:<10}  L={g['left_hamming']:>3} R={g['right_hamming']:>3}  "
            f"before/after"
        )
        label_strip = _label_row(label, row_img.shape[1])
        row_full = np.concatenate([label_strip, row_img], axis=0)
        rows.append(row_full)
        rows.append(np.zeros((ROW_GAP, row_full.shape[1], 3), dtype=np.uint8))

    cap.release()
    if not rows:
        return None
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)
    return np.concatenate(padded, axis=0)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for vid in TARGET_VIDEOS:
        print(f"=== video_{vid} viz 生成中 ===")
        mosaic = build_video_mosaic(vid)
        if mosaic is None:
            print(f"  [SKIP] video_{vid}")
            continue
        out_path = OUT_DIR / f"win_panel_probe_c_{vid}_2026-07-23.png"
        cv2.imwrite(str(out_path), mosaic)
        print(f"  保存: {out_path}  shape={mosaic.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
