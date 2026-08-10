"""v1/v2 fine-tune model の検証 (2026-08-07)。

検証(a): 問題領域 = dio_vs_ts_m01_clip.mp4 t=313.00s (frame=9390) の
1P 上部 (row1-4, 全列24セル)。warm_full_t313s.png で目視確認済みの
「暗赤系キャラ背景が puyo 色に誤読される」領域。ゲームルール上、
この帯は当該フレームで盤面が埋まっておらず可視スプライトが存在しない
= 空セルであることが画像目視で確定している (ground_truth=EMPTY)。

検証(b): 実ぷよセルのスポット正解維持。既存5動画 (v50/v70/v89m3/v91/v97)
の擬似ラベル (実ぷよ色ラベル付き、色0以外) から固定 seed でランダム
サンプルし、v1/v2 で正解率を比較する。

本番モデル・本番設定には一切触れない使い捨てスクリプト。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, COLOR_EMPTY
from src.image_reader import DEFAULT_P1_REGION
from src.patch_classifier import CnnPatchClassifierLarge
from src.self_supervised.label_store import LabelStore
from src.self_supervised.pseudo_label import COMPONENT_CELL

CLIP_PATH = Path("data/verify/youtube_demo_2026-08-07/dio_vs_ts_warmup_clip.mp4")
PROBLEM_T_SEC = 313.00
PROBLEM_ROWS = (1, 2, 3, 4)  # warm_full_t313s.png で目視確認済み・空確定

STORE_ROOT = Path("data/pseudo_labels_olRyxDGacbg_demo_2026-08-07")
SPOT_VIDEO_IDS = ("v50", "v70", "v89m3", "v91", "v97")
SPOT_N_SAMPLES = 300
SPOT_SEED = 42

MODEL_V1 = Path("models/cnn_finetune_olRyxDGacbg_demo_2026-08-07.pt")
MODEL_V2 = Path("models/cnn_finetune_olRyxDGacbg_demo_v2_2026-08-07.pt")

TARGET_SIZE = (1920, 1080)


def _load_model(path: Path) -> CnnPatchClassifierLarge:
    cnn = CnnPatchClassifierLarge()
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    cnn._model.load_state_dict(state)
    return cnn


def _extract_patch(frame: np.ndarray, region, row: int, col: int) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(y1 + 1, min(int(y2), h))
    patch = frame[y1:y2, x1:x2]
    return patch.copy() if patch.size > 0 else None


def _collect_problem_patches() -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(CLIP_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {CLIP_PATH}")
    cap.set(cv2.CAP_PROP_POS_MSEC, PROBLEM_T_SEC * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read frame at t={PROBLEM_T_SEC}s")
    if frame.shape[1::-1] != TARGET_SIZE:
        frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    patches = []
    for row in PROBLEM_ROWS:
        for col in range(BOARD_COLS):
            patch = _extract_patch(frame, DEFAULT_P1_REGION, row, col)
            if patch is not None:
                patches.append(patch)
    return patches


def _eval_problem_region(cnn: CnnPatchClassifierLarge, patches: list[np.ndarray]) -> tuple[int, int]:
    n_correct = 0
    for patch in patches:
        pred = cnn.classify(patch)
        if pred == COLOR_EMPTY:
            n_correct += 1
    return n_correct, len(patches)


def _collect_spot_samples() -> list[tuple[np.ndarray, int]]:
    """既存5動画の実ぷよ (color != EMPTY) seed から固定 seed でサンプル."""
    all_real: list[tuple[np.ndarray, int]] = []
    for vid in SPOT_VIDEO_IDS:
        store = LabelStore(video_id=vid, root=STORE_ROOT)
        for s in store.load(COMPONENT_CELL):
            if int(s.label) != COLOR_EMPTY:
                all_real.append((s.input_data["patch"], int(s.label)))
    rng = random.Random(SPOT_SEED)
    rng.shuffle(all_real)
    return all_real[:SPOT_N_SAMPLES]


def _eval_spot(cnn: CnnPatchClassifierLarge, samples: list[tuple[np.ndarray, int]]) -> tuple[int, int]:
    n_correct = 0
    for patch, label in samples:
        pred = cnn.classify(patch)
        if pred == label:
            n_correct += 1
    return n_correct, len(samples)


def main() -> int:
    print("[verify] loading models...")
    cnn_v1 = _load_model(MODEL_V1)
    cnn_v2 = _load_model(MODEL_V2)

    print(f"[verify] (a) problem region: {CLIP_PATH.name} t={PROBLEM_T_SEC}s "
          f"1P row{PROBLEM_ROWS} x {BOARD_COLS}cols")
    problem_patches = _collect_problem_patches()
    n1, d1 = _eval_problem_region(cnn_v1, problem_patches)
    n2, d2 = _eval_problem_region(cnn_v2, problem_patches)
    print(f"[verify] (a) v1 empty-acc = {n1}/{d1} = {100.0*n1/d1:.1f}%")
    print(f"[verify] (a) v2 empty-acc = {n2}/{d2} = {100.0*n2/d2:.1f}%")

    print(f"[verify] (b) spot check: {SPOT_N_SAMPLES} real-puyo cells "
          f"from {SPOT_VIDEO_IDS}")
    spot_samples = _collect_spot_samples()
    m1, e1 = _eval_spot(cnn_v1, spot_samples)
    m2, e2 = _eval_spot(cnn_v2, spot_samples)
    print(f"[verify] (b) v1 real-acc = {m1}/{e1} = {100.0*m1/e1:.1f}%")
    print(f"[verify] (b) v2 real-acc = {m2}/{e2} = {100.0*m2/e2:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
