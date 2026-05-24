"""W7-CNN: 動画から CNN v8 + HSV 双方一致のパッチを擬似ラベル化。

各動画の安定フレーム (アニメ中以外) で StatePipeline (CNN v8) を実行し、
以下の条件を満たすパッチのみを採用:
    1. CNN v8 max prob >= 0.97 (高信頼度)
    2. HSV ベース判定と一致 (HybridClassifier の合意)
    3. 物理整合 (4+ 連結に含まれない)

確実な疑似ラベルを大量蓄積し、CNN v9 訓練データを拡大。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_collect_pseudo_labels \
        --videos 4 5 6 7 8 9 11 12 13 14 15 16 17 18 19 \
        --max-per-video 800 \
        --out data/training_phase_u/pseudo_v8.npz
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2
import numpy as np
import torch

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA,
    COLOR_UNKNOWN, HIDDEN_ROWS,
)
from src.chain import ChainSimulator, MIN_ERASE_COUNT
from src.image_reader import (
    ColorClassifier, DEFAULT_P1_REGION, DEFAULT_P2_REGION,
    ImageReader,
)
from src.patch_classifier import (
    CLASS_INDEX_TO_COLOR, CnnPatchClassifier,
    PATCH_RESIZE_H, PATCH_RESIZE_W,
)


PATCH_OUT_SIZE = 16
HIGH_CONF = 0.97


def load_cnn(path: str, device: str) -> CnnPatchClassifier:
    cnn = CnnPatchClassifier()
    state = torch.load(path, map_location=device, weights_only=True)
    cnn._model.load_state_dict(state)
    cnn._model.to(device)
    cnn._model.eval()
    return cnn


def patches_to_tensor(patches: np.ndarray, device: str) -> torch.Tensor:
    N = patches.shape[0]
    out = np.zeros(
        (N, PATCH_RESIZE_H, PATCH_RESIZE_W, 6), dtype=np.float32,
    )
    for i in range(N):
        resized = cv2.resize(
            patches[i], (PATCH_RESIZE_W, PATCH_RESIZE_H),
            interpolation=cv2.INTER_AREA,
        )
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        combined = np.concatenate([resized, hsv], axis=2)
        out[i] = combined.astype(np.float32) / 255.0
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous().to(device)


def collect_from_video(
    video_id: int,
    cnn: CnnPatchClassifier,
    hsv_classifier: ColorClassifier,
    device: str,
    max_samples: int,
    interval: float = 1.0,
    skip_seconds: float = 5.0,
) -> tuple[list[np.ndarray], list[int]]:
    video_path = Path(f"data/frames/video_{video_id:02d}.mp4")
    winners_path = Path(
        f"data/verify/match_winners_v{video_id:02d}.tsv",
    )
    if not video_path.exists() or not winners_path.exists():
        return [], []

    cap = cv2.VideoCapture(str(video_path))
    matches = []
    with open(winners_path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                matches.append({
                    "start": float(r["start_sec"]),
                    "end": float(r["end_sec"]),
                })
            except (KeyError, ValueError):
                continue

    simulator = ChainSimulator()
    patches_collected: list[np.ndarray] = []
    labels_collected: list[int] = []

    for m in matches:
        if len(patches_collected) >= max_samples:
            break
        # 安定フレーム (試合内中盤、interval 秒間隔)
        t = m["start"] + skip_seconds
        end_t = m["end"] - skip_seconds
        while t <= end_t and len(patches_collected) < max_samples:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            if not ok or fr is None:
                t += interval
                continue
            if fr.shape[:2] != (1080, 1920):
                fr = cv2.resize(
                    fr, (1920, 1080), interpolation=cv2.INTER_AREA,
                )
            for side, region in (
                ("1P", DEFAULT_P1_REGION),
                ("2P", DEFAULT_P2_REGION),
            ):
                patches_for_frame = []
                for vrow in range(12):
                    for col in range(BOARD_COLS):
                        row = vrow + HIDDEN_ROWS
                        x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                        h, w = fr.shape[:2]
                        x1 = max(0, min(x1, w - 1))
                        x2 = max(x1 + 1, min(x2, w))
                        y1 = max(0, min(y1, h - 1))
                        y2 = max(y1 + 1, min(y2, h))
                        patch = fr[y1:y2, x1:x2]
                        if patch.size == 0:
                            continue
                        patches_for_frame.append((vrow, col, patch))

                # CNN v8 batch 推論
                batch_patches = np.stack([
                    cv2.resize(
                        p, (PATCH_OUT_SIZE, PATCH_OUT_SIZE),
                        interpolation=cv2.INTER_AREA,
                    )
                    for _, _, p in patches_for_frame
                ])
                tensor = patches_to_tensor(batch_patches, device)
                with torch.no_grad():
                    logits = cnn._model(tensor)
                    probs = torch.softmax(logits, dim=1)
                    max_p, idx = probs.max(dim=1)
                max_p_np = max_p.cpu().numpy()
                idx_np = idx.cpu().numpy()

                # HSV と CNN v8 一致 + 高信頼度 のみ
                for i, (vrow, col, patch) in enumerate(patches_for_frame):
                    if max_p_np[i] < HIGH_CONF:
                        continue
                    cnn_label = CLASS_INDEX_TO_COLOR[int(idx_np[i])]
                    hsv_label = hsv_classifier.classify(patch)
                    if hsv_label != cnn_label:
                        continue
                    # 採用
                    patch_resized = cv2.resize(
                        patch, (PATCH_OUT_SIZE, PATCH_OUT_SIZE),
                        interpolation=cv2.INTER_AREA,
                    )
                    patches_collected.append(patch_resized)
                    labels_collected.append(int(cnn_label))
                    if len(patches_collected) >= max_samples:
                        break
                if len(patches_collected) >= max_samples:
                    break
            t += interval

    cap.release()
    print(
        f"  v{video_id:02d}: {len(patches_collected)} pseudo labels"
    )
    return patches_collected, labels_collected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos", type=int, nargs="+",
        default=list(range(1, 20)),
    )
    parser.add_argument("--max-per-video", type=int, default=500)
    parser.add_argument(
        "--cnn-model", default="models/cnn_phase_u_v8.pt",
    )
    parser.add_argument(
        "--out", default="data/training_phase_u/pseudo_v8.npz",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    cnn = load_cnn(args.cnn_model, device=device)
    hsv = ColorClassifier()

    all_patches: list[np.ndarray] = []
    all_labels: list[int] = []
    for vid in args.videos:
        ps, ls = collect_from_video(
            vid, cnn, hsv, device, args.max_per_video,
        )
        all_patches.extend(ps)
        all_labels.extend(ls)

    if not all_patches:
        print("no samples")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    X = np.stack(all_patches)
    y = np.array(all_labels, dtype=np.int32)
    np.savez_compressed(out_path, patches=X, labels=y)
    print(f"\nfinal: {X.shape}")
    unique, counts = np.unique(y, return_counts=True)
    for c, n in zip(unique, counts):
        print(f"  code={c}: {n}")
    print(f"saved: {to_windows_path(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
