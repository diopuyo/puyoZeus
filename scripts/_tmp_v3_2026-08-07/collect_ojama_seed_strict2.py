"""ojama seed 採取 v3 (strict2): eye-pupil (暗色2点) シグネチャを追加要求。

strict (v3_strict) でも汚染 ~20-30% 残存 (バースト半透明オーバーレイ・UI破片が
低彩度高輝度で一致してしまう、reference_burst_overlay_semitransparent_2026-08-05
の「バースト=5色の半透明レイヤー」と整合)。

追加フィルタ: 本物の ojama スプライトは中央付近に暗い瞳 (V<90) が
小面積 (3-20%) で存在し、 残りは明るい灰色本体 (S<16 & V>=120, >=55%)。
単色べったりの汚染 (バースト overlay・UI破片) はこの二値構造を持たない
ことが多いため、 この bimodal 判定を追加して汚染を減らす。
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
OUT_VIDEO_ID = "olRyxDGacbg_ojama_seed_v3_strict2"

STEP_SEC: float = 1.5
EDGE_MARGIN_SEC: float = 1.0

ROW_START = HIDDEN_ROWS
ROW_END_EXCLUSIVE = HIDDEN_ROWS + 12

STRICT_S_MAX: int = 16
STRICT_V_MIN: int = 120
STRICT_PIXEL_RATIO_MIN: float = 0.55  # 瞳を除いた本体分を考慮し少し緩和
CENTER_CROP_RATIO: float = 0.8

# 瞳シグネチャ判定 (bimodal 構造要求)
EYE_V_MAX: int = 90
EYE_AREA_RATIO_MIN: float = 0.02
EYE_AREA_RATIO_MAX: float = 0.22


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


def _passes_strict2(patch: np.ndarray) -> bool:
    h, w = patch.shape[:2]
    m = (1.0 - CENTER_CROP_RATIO) / 2.0
    crop = patch[int(h * m):int(h * (1 - m)), int(w * m):int(w * (1 - m))]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    body_mask = (s < STRICT_S_MAX) & (v >= STRICT_V_MIN)
    body_ratio = float(body_mask.mean())
    if body_ratio < STRICT_PIXEL_RATIO_MIN:
        return False
    eye_mask = v < EYE_V_MAX
    eye_ratio = float(eye_mask.mean())
    if not (EYE_AREA_RATIO_MIN <= eye_ratio <= EYE_AREA_RATIO_MAX):
        return False
    # 本体+瞳で大半を説明できるか (その他の色混入が少ない = 汚染低減)
    if body_ratio + eye_ratio < 0.75:
        return False
    return True


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
            if not _passes_strict2(patch):
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
                    "ground_truth_rule": "hsv_median_ojama_AND_body55pct_AND_eye_bimodal",
                },
            ))
    return out


def main() -> int:
    matches = _read_matches(MATCHES_TSV)
    print(f"[collect_ojama_strict2] {len(matches)} matches, step={STEP_SEC}s")

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
        print(f"[collect_ojama_strict2] match={match_idx} ({start_sec:.0f}-{end_sec:.0f}s) "
              f"frames={n_frames_this_match} cum_total={total}")
    cap.release()
    print(f"[collect_ojama_strict2] DONE total={total} -> {store.video_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
