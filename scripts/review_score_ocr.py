"""score OCR の動作を目視レビューする画像を生成する。

複数の試合・複数の時刻でフレームを取得し、
- score ROI (1P / 2P) を切り出し
- OCR で読み取った値と confidence
- 数字の桁ごとの読取 (None=失敗)
を並べた grid 画像を出力する。

出力:
    data/verify/review_score_ocr_grid.png
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.score_ocr import (
    DIGIT_HEIGHT,
    DIGIT_LEFTS_1P,
    DIGIT_TOP,
    DIGIT_WIDTH,
    EXPECTED_FRAME_SHAPE,
    SCORE_1P_REGION,
    SCORE_2P_REGION,
    ScoreOcr,
)

OUT_PATH = Path("data/verify/review_score_ocr_grid.png")
ROI_HEIGHT = SCORE_1P_REGION[1] - SCORE_1P_REGION[0]  # 65
ROI_WIDTH = SCORE_1P_REGION[3] - SCORE_1P_REGION[2]   # 320

# レビューする (動画, 試合 idx, 試合内オフセット秒) のリスト
REVIEW_TARGETS: list[tuple[str, int, float]] = [
    ("video_01", 1, 5.0),    # 試合中盤
    ("video_01", 1, 30.0),   # 試合中盤
    ("video_01", 1, 60.0),   # 試合終盤
    ("video_01", 5, 20.0),
    ("video_02", 1, 30.0),   # 720p
    ("video_02", 1, 50.0),
    ("video_02", 25, 30.0),
    ("video_02", 50, 20.0),
    ("video_03", 1, 30.0),
    ("video_03", 20, 30.0),
]


def fetch_frame(
    video_path: Path, t_sec: float
) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != EXPECTED_FRAME_SHAPE:
        frame = cv2.resize(
            frame,
            (EXPECTED_FRAME_SHAPE[1], EXPECTED_FRAME_SHAPE[0]),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def annotate_panel(
    panel: np.ndarray,
    title: str,
    score: int | None,
    confidence: float,
    digits: tuple[int | None, ...],
) -> np.ndarray:
    """1 試合分のパネル (1P + 2P ROI) にタイトルと OCR 結果を描き込む。"""
    h, w = panel.shape[:2]
    out = np.zeros((h + 60, w, 3), dtype=np.uint8)
    out[60:, :, :] = panel
    cv2.putText(out, title, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if score is None:
        text = f"OCR FAIL  conf={confidence:.2f}  digits={digits}"
        color = (0, 0, 255)
    else:
        text = f"score={score}  conf={confidence:.2f}"
        color = (0, 255, 0)
    cv2.putText(out, text, (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # digits を桁ごとに細かく
    digit_text = "  ".join(
        ("?" if d is None else str(d)) for d in digits
    )
    cv2.putText(out, f"d:[{digit_text}]", (8, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return out


def build_row(
    frame: np.ndarray,
    label: str,
    ocr: ScoreOcr,
) -> np.ndarray:
    """1 フレーム分の結果行 (1P ROI + 2P ROI を横に並べたパネル) を作る。"""
    res = ocr.read(frame)
    y1, y2, x1, x2 = SCORE_1P_REGION
    roi_1p = frame[y1:y2, x1:x2].copy()
    y1, y2, x1, x2 = SCORE_2P_REGION
    roi_2p = frame[y1:y2, x1:x2].copy()

    # ROI に桁の境界線を描く (デバッグ用)
    for x in DIGIT_LEFTS_1P:
        cv2.line(roi_1p, (x, 0), (x, ROI_HEIGHT), (60, 60, 60), 1)
        cv2.line(roi_2p, (x, 0), (x, ROI_HEIGHT), (60, 60, 60), 1)

    panel_1p = annotate_panel(
        roi_1p, f"{label} | 1P",
        res.score_1p, res.confidence_1p, res.digits_1p,
    )
    panel_2p = annotate_panel(
        roi_2p, f"{label} | 2P",
        res.score_2p, res.confidence_2p, res.digits_2p,
    )
    # 横に結合 (細い区切り線挿入)
    sep = np.full((panel_1p.shape[0], 4, 3), 80, dtype=np.uint8)
    return np.hstack([panel_1p, sep, panel_2p])


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ocr = ScoreOcr.load_default()

    rows: list[np.ndarray] = []
    for vid, idx, offset in REVIEW_TARGETS:
        # matches.tsv から試合の start_sec を取得
        bdy_path = Path(f"data/verify/match_boundaries_v4/{vid}/matches.tsv")
        if not bdy_path.is_file():
            print(f"[skip] boundary なし: {bdy_path}")
            continue
        with open(bdy_path) as f:
            match_rows = list(csv.DictReader(f, delimiter="\t"))
        match = next((m for m in match_rows if int(m["idx"]) == idx), None)
        if match is None:
            print(f"[skip] {vid} 試合 {idx} なし")
            continue
        start = float(match["start_sec"])
        t = start + offset
        video_path = Path(f"data/frames/{vid}.mp4")
        frame = fetch_frame(video_path, t)
        if frame is None:
            print(f"[skip] フレーム取得失敗: {vid} t={t}")
            continue
        label = f"{vid} match{idx} t={t:.1f}s (offset+{offset:.0f}s)"
        row_panel = build_row(frame, label, ocr)
        rows.append(row_panel)
        print(f"  [ok] {label}")

    if not rows:
        print("レビュー対象なし")
        return 1
    # 全行を縦結合
    sep = np.full((6, rows[0].shape[1], 3), 40, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for r in rows:
        parts.append(r)
        parts.append(sep)
    grid = np.vstack(parts[:-1])

    cv2.imwrite(str(OUT_PATH), grid)
    print(f"\n出力: {OUT_PATH} (shape={grid.shape})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
