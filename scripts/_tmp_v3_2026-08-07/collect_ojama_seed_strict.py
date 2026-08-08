"""video_olRyxDGacbg から ojama seed をHSVルール(厳しめ)で採取する v2。

第一回 (collect_ojama_seed.py, 3460枚) のモンタージュ目視で ~15-20% の
汚染 (puyo ハイライト縁・背景・UI破片) を確認。原因はデフォルトの
classify() が中央値 (median) 判定のため、 セル半分が puyo ハイライトの
白っぽい領域でも中央値が低彩度側に落ちて OJAMA 誤判定するケース。

対策: 中央 80% クロップの pixel-wise 投票で「低彩度 (S<16) かつ高輝度
(V>=120)」画素が 65% 以上を占める場合のみ採用する (median 一発判定より
厳格)。ColorClassifier 本体 (src/image_reader.py) は変更しない
(既存 vote_mode ロジックとは独立の使い捨てフィルタ)。

出力: data/pseudo_labels_olRyxDGacbg_demo_2026-08-07/olRyxDGacbg_ojama_seed_v3_strict/cell.jsonl
(v3 (非strict) ディレクトリは残置、strict の方を最終的に fine-tune に使う)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, COLOR_OJAMA, HIDDEN_ROWS
from src.image_reader import ColorClassifier, DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.self_supervised.label_store import LabelStore
from src.self_supervised.pseudo_label import COMPONENT_CELL, PseudoLabelSample

VIDEO_PATH = _ROOT / "data/frames/video_olRyxDGacbg.mp4"
MATCHES_TSV = _ROOT / "data/verify/match_boundaries_olRyxDGacbg/video_olRyxDGacbg/matches.tsv"
OUT_ROOT = _ROOT / "data/pseudo_labels_olRyxDGacbg_demo_2026-08-07"
OUT_VIDEO_ID = "olRyxDGacbg_ojama_seed_v3_strict"

STEP_SEC: float = 1.5
EDGE_MARGIN_SEC: float = 1.0

ROW_START = HIDDEN_ROWS
ROW_END_EXCLUSIVE = HIDDEN_ROWS + 12

# 厳格フィルタ閾値 (median 判定より狭い margin + pixel-wise 多数決)
STRICT_S_MAX: int = 16
STRICT_V_MIN: int = 120
STRICT_PIXEL_RATIO_MIN: float = 0.65
CENTER_CROP_RATIO: float = 0.8  # 中央 80% のみ評価(縁の背景混入を回避)


def _read_matches(path: Path) -> list[tuple[float, float]]:
    out = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out.append((float(row["start_sec"]), float(row["end_sec"])))
    return out


def _extract_patch(frame: np.ndarray, region, row: int, col: int) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(y1 + 1, min(int(y2), h))
    patch = frame[y1:y2, x1:x2]
    return patch.copy() if patch.size > 0 else None


def _passes_strict_pixel_vote(patch: np.ndarray) -> bool:
    h, w = patch.shape[:2]
    m = (1.0 - CENTER_CROP_RATIO) / 2.0
    crop = patch[int(h * m):int(h * (1 - m)), int(w * m):int(w * (1 - m))]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = (s < STRICT_S_MAX) & (v >= STRICT_V_MIN)
    ratio = float(mask.mean())
    return ratio >= STRICT_PIXEL_RATIO_MIN


def _collect_frame(
    hsv_clf: ColorClassifier, frame: np.ndarray, region, side: str,
    match_idx: int, t_sec: float,
) -> list[PseudoLabelSample]:
    out: list[PseudoLabelSample] = []
    for row in range(ROW_START, ROW_END_EXCLUSIVE):
        for col in range(BOARD_COLS):
            patch = _extract_patch(frame, region, row, col)
            if patch is None:
                continue
            pred = hsv_clf.classify(patch)
            if pred != COLOR_OJAMA:
                continue
            if not _passes_strict_pixel_vote(patch):
                continue
            out.append(PseudoLabelSample(
                component=COMPONENT_CELL,
                timestamp=t_sec,
                input_data={"patch": patch},
                label=COLOR_OJAMA,
                confidence=1.0,
                metadata={
                    "video_id": OUT_VIDEO_ID,
                    "match_idx": match_idx,
                    "row": row, "col": col, "side": side,
                    "ground_truth_rule": "hsv_median_ojama_AND_strict_pixel_vote_65pct",
                },
            ))
    return out


def main() -> int:
    matches = _read_matches(MATCHES_TSV)
    print(f"[collect_ojama_strict] {len(matches)} matches, step={STEP_SEC}s")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {VIDEO_PATH}")

    hsv_clf = ColorClassifier()
    store = LabelStore(video_id=OUT_VIDEO_ID, root=OUT_ROOT)
    total = 0
    for match_idx, (start_sec, end_sec) in enumerate(matches, start=1):
        t = start_sec + EDGE_MARGIN_SEC
        end = end_sec - EDGE_MARGIN_SEC
        n_frames_this_match = 0
        while t <= end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                t += STEP_SEC
                continue
            s1 = _collect_frame(hsv_clf, frame, DEFAULT_P1_REGION, "1P", match_idx, t)
            s2 = _collect_frame(hsv_clf, frame, DEFAULT_P2_REGION, "2P", match_idx, t)
            samples = s1 + s2
            if samples:
                store.append(samples)
                total += len(samples)
            n_frames_this_match += 1
            t += STEP_SEC
        print(f"[collect_ojama_strict] match={match_idx} ({start_sec:.0f}-{end_sec:.0f}s) "
              f"frames={n_frames_this_match} cum_total={total}")
    cap.release()
    print(f"[collect_ojama_strict] DONE total={total} -> {store.video_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
