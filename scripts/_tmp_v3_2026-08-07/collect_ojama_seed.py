"""video_olRyxDGacbg から ojama seed をHSVルールで採取する (v3 fine-tune用、使い捨て)。

根拠: src/image_reader.py の ColorClassifier.classify() は
S < OJAMA_S_THRESHOLD(20) and V >= OJAMA_V_MIN(100) で OJAMA 判定する
既存ルールベース実装 (CNN 診断とは独立)。診断 (_tmp_v3_2026-08-07/diag_t2926.py)
で base/v1/v2 の3モデル全てが誤判定した5セル全てを、この HSV ルールは
正しく OJAMA と判定できていた (t=2926s実測)。このルールを使い、
matches.tsv の全試合区間から ojama patch を機械的に収集する。

出力: data/pseudo_labels_olRyxDGacbg_demo_2026-08-07/olRyxDGacbg_ojama_seed_v3/cell.jsonl
本番モデル・既存 seed には一切触れない。
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
OUT_VIDEO_ID = "olRyxDGacbg_ojama_seed_v3"

STEP_SEC: float = 1.5  # 試合区間内のサンプリング間隔
EDGE_MARGIN_SEC: float = 1.0  # 試合開始/終了直後の遷移ノイズを避ける

ROW_START = HIDDEN_ROWS  # 1
ROW_END_EXCLUSIVE = HIDDEN_ROWS + 12  # 13 (可視12行)


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
                    "ground_truth_rule": "hsv_s_lt_20_v_ge_100",
                },
            ))
    return out


def main() -> int:
    matches = _read_matches(MATCHES_TSV)
    print(f"[collect_ojama] {len(matches)} matches, step={STEP_SEC}s")

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
        print(f"[collect_ojama] match={match_idx} ({start_sec:.0f}-{end_sec:.0f}s) "
              f"frames={n_frames_this_match} cum_total={total}")
    cap.release()
    print(f"[collect_ojama] DONE total={total} -> {store.video_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
