"""t=2926s (video_olRyxDGacbg) 1P盤面の全セルを base/v1/v2 の3モデルで
classify() 比較する診断スクリプト (使い捨て、本番資産は一切変更しない)。

出力: 各セル (row 1-12, col 1-6, 1-indexed) の HSV rule 判定 + 3モデル予測。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, COLOR_EMPTY, COLOR_OJAMA, HIDDEN_ROWS
from src.image_reader import ColorClassifier, DEFAULT_P1_REGION
from src.patch_classifier import CnnPatchClassifierLarge

VIDEO_PATH = _ROOT / "data/frames/video_olRyxDGacbg.mp4"
T_SEC = 2926.0
TARGET_SIZE = (1920, 1080)

MODEL_BASE = _ROOT / "models/cnn_phase_i_hsv_seed.pt"
MODEL_V1 = _ROOT / "models/cnn_finetune_olRyxDGacbg_demo_2026-08-07.pt"
MODEL_V2 = _ROOT / "models/cnn_finetune_olRyxDGacbg_demo_v2_2026-08-07.pt"

COLOR_NAME = {0: "EMPTY", 1: "RED", 2: "BLUE", 3: "GREEN", 4: "YELLOW", 5: "PURPLE", 9: "OJAMA"}


def load_model(path: Path) -> CnnPatchClassifierLarge:
    cnn = CnnPatchClassifierLarge()
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    cnn._model.load_state_dict(state)
    return cnn


def extract_patch(frame: np.ndarray, region, row: int, col: int) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(y1 + 1, min(int(y2), h))
    patch = frame[y1:y2, x1:x2]
    return patch.copy() if patch.size > 0 else None


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    cap.set(cv2.CAP_PROP_POS_MSEC, T_SEC * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("read failed")
    if frame.shape[1::-1] != TARGET_SIZE:
        frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    hsv_clf = ColorClassifier()
    cnn_base = load_model(MODEL_BASE)
    cnn_v1 = load_model(MODEL_V1)
    cnn_v2 = load_model(MODEL_V2)

    region = DEFAULT_P1_REGION
    print(f"[diag] t={T_SEC}s 1P board, row(内部1-12,可視) x col(内部0-5)")
    print(f"{'row':>4} {'col':>4} {'hsv':>8} {'base':>8} {'v1':>8} {'v2':>8}")
    n_hsv_ojama = 0
    hsv_ojama_cells = []
    for row in range(HIDDEN_ROWS, HIDDEN_ROWS + 12):
        for col in range(BOARD_COLS):
            patch = extract_patch(frame, region, row, col)
            if patch is None:
                continue
            hsv_pred = hsv_clf.classify(patch)
            base_pred = cnn_base.classify(patch)
            v1_pred = cnn_v1.classify(patch)
            v2_pred = cnn_v2.classify(patch)
            if hsv_pred != COLOR_EMPTY or base_pred != COLOR_EMPTY or v2_pred != COLOR_EMPTY:
                print(f"{row:>4} {col:>4} {COLOR_NAME.get(hsv_pred,hsv_pred):>8} "
                      f"{COLOR_NAME.get(base_pred,base_pred):>8} "
                      f"{COLOR_NAME.get(v1_pred,v1_pred):>8} "
                      f"{COLOR_NAME.get(v2_pred,v2_pred):>8}")
            if hsv_pred == COLOR_OJAMA:
                n_hsv_ojama += 1
                hsv_ojama_cells.append((row, col))
    print(f"[diag] HSV rule OJAMA判定セル数={n_hsv_ojama}: {hsv_ojama_cells}")


if __name__ == "__main__":
    main()
