"""v2/v3 fine-tune model の検証 (2026-08-07、v3用)。

検証(a): t=2926s (FINAL2 clip t=51s) 1P おじゃまセル 5個の正解率。
検証(b): t=2899s (FINAL2 clip t=24s) 1P おじゃまセル 8個 (右3列サブセット3個) の正解率。
検証(c): warmup_clip t=313.00s 1P上部 (row1-4) 空セル正解率 (v2 baseline 95.8%回帰確認)。
検証(d): 既存5動画 実ぷよ300セルスポット正解率維持確認。

ground truth の (a)(b) セル座標は _tmp_v3_2026-08-07/diag_t2926.py,
diag_t2899.py の HSV ルール判定 (目視確認済み、diag出力を参照) から採用。

本番モデル・本番設定には一切触れない使い捨てスクリプト。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, COLOR_EMPTY, COLOR_OJAMA
from src.image_reader import DEFAULT_P1_REGION
from src.patch_classifier import CnnPatchClassifierLarge
from src.self_supervised.label_store import LabelStore
from src.self_supervised.pseudo_label import COMPONENT_CELL

VIDEO_PATH = _ROOT / "data/frames/video_olRyxDGacbg.mp4"
WARMUP_CLIP = _ROOT / "data/verify/youtube_demo_2026-08-07/dio_vs_ts_warmup_clip.mp4"
TARGET_SIZE = (1920, 1080)

T_2926 = 2926.0
T_2926_OJAMA_CELLS = [(3, 3), (5, 1), (5, 4), (6, 3), (10, 5)]

T_2899 = 2899.0
T_2899_OJAMA_CELLS = [
    (3, 0), (4, 0), (4, 2), (4, 4), (5, 0), (5, 1), (5, 3), (5, 5),
]
T_2899_RIGHT3_CELLS = [(4, 4), (5, 3), (5, 5)]

PROBLEM_T_SEC = 313.00
PROBLEM_ROWS = (1, 2, 3, 4)

STORE_ROOT = _ROOT / "data/pseudo_labels_olRyxDGacbg_demo_2026-08-07"
SPOT_VIDEO_IDS = ("v50", "v70", "v89m3", "v91", "v97")
SPOT_N_SAMPLES = 300
SPOT_SEED = 42

MODEL_BASE = _ROOT / "models/cnn_phase_i_hsv_seed.pt"
MODEL_V2 = _ROOT / "models/cnn_finetune_olRyxDGacbg_demo_v2_2026-08-07.pt"
MODEL_V3 = _ROOT / "models/cnn_finetune_olRyxDGacbg_demo_v3_2026-08-07.pt"


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


def read_frame(video_path: Path, t_sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read frame at t={t_sec}s")
    if frame.shape[1::-1] != TARGET_SIZE:
        frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return frame


def eval_ojama_cells(
    cnn: CnnPatchClassifierLarge, frame: np.ndarray, cells: list[tuple[int, int]],
) -> tuple[int, int]:
    n_correct = 0
    for row, col in cells:
        patch = extract_patch(frame, DEFAULT_P1_REGION, row, col)
        if patch is None:
            continue
        pred = cnn.classify(patch)
        if pred == COLOR_OJAMA:
            n_correct += 1
    return n_correct, len(cells)


def eval_empty_region(
    cnn: CnnPatchClassifierLarge, frame: np.ndarray,
) -> tuple[int, int]:
    n_correct, n_total = 0, 0
    for row in PROBLEM_ROWS:
        for col in range(BOARD_COLS):
            patch = extract_patch(frame, DEFAULT_P1_REGION, row, col)
            if patch is None:
                continue
            n_total += 1
            if cnn.classify(patch) == COLOR_EMPTY:
                n_correct += 1
    return n_correct, n_total


def collect_spot_samples() -> list[tuple[np.ndarray, int]]:
    all_real: list[tuple[np.ndarray, int]] = []
    for vid in SPOT_VIDEO_IDS:
        store = LabelStore(video_id=vid, root=STORE_ROOT)
        for s in store.load(COMPONENT_CELL):
            if int(s.label) != COLOR_EMPTY:
                all_real.append((s.input_data["patch"], int(s.label)))
    rng = random.Random(SPOT_SEED)
    rng.shuffle(all_real)
    return all_real[:SPOT_N_SAMPLES]


def eval_spot(
    cnn: CnnPatchClassifierLarge, samples: list[tuple[np.ndarray, int]],
) -> tuple[int, int]:
    n_correct = 0
    for patch, label in samples:
        if cnn.classify(patch) == label:
            n_correct += 1
    return n_correct, len(samples)


def main() -> int:
    print("[verify_v3] loading models...")
    models = {
        "base": load_model(MODEL_BASE),
        "v2": load_model(MODEL_V2),
        "v3": load_model(MODEL_V3),
    }

    frame_2926 = read_frame(VIDEO_PATH, T_2926)
    frame_2899 = read_frame(VIDEO_PATH, T_2899)
    spot_samples = collect_spot_samples()
    warmup_frame = None
    if WARMUP_CLIP.is_file():
        warmup_frame = read_frame(WARMUP_CLIP, PROBLEM_T_SEC)

    print(f"\n[verify_v3] (a) t={T_2926}s ojama cells n={len(T_2926_OJAMA_CELLS)}")
    for name, cnn in models.items():
        n, d = eval_ojama_cells(cnn, frame_2926, T_2926_OJAMA_CELLS)
        print(f"  {name}: {n}/{d} = {100.0 * n / d:.1f}%")

    print(f"\n[verify_v3] (b1) t={T_2899}s ojama cells (全体) n={len(T_2899_OJAMA_CELLS)}")
    for name, cnn in models.items():
        n, d = eval_ojama_cells(cnn, frame_2899, T_2899_OJAMA_CELLS)
        print(f"  {name}: {n}/{d} = {100.0 * n / d:.1f}%")

    print(f"\n[verify_v3] (b2) t={T_2899}s ojama cells (右3列4-6列目) n={len(T_2899_RIGHT3_CELLS)}")
    for name, cnn in models.items():
        n, d = eval_ojama_cells(cnn, frame_2899, T_2899_RIGHT3_CELLS)
        print(f"  {name}: {n}/{d} = {100.0 * n / d:.1f}%")

    if warmup_frame is not None:
        print(f"\n[verify_v3] (c) warmup t={PROBLEM_T_SEC}s empty region (row1-4 x 6col)")
        for name, cnn in models.items():
            n, d = eval_empty_region(cnn, warmup_frame)
            print(f"  {name}: {n}/{d} = {100.0 * n / d:.1f}%")
    else:
        print(f"\n[verify_v3] (c) SKIP: warmup clip not found at {WARMUP_CLIP}")

    print(f"\n[verify_v3] (d) 実ぷよ{len(spot_samples)}セル スポット正解率")
    for name, cnn in models.items():
        n, d = eval_spot(cnn, spot_samples)
        print(f"  {name}: {n}/{d} = {100.0 * n / d:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
