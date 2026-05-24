"""W14-B: CNN v15 - v14 init から review-only fine-tune。

戦略:
    1. v14 (general training 済) を init として load
    2. review labels 2700 cells のみで 8 epoch fine-tune
    3. class balance なし、低 LR (1e-4) で慎重に
    4. v18_m03 のような難 cell を直接学習させる

期待:
    v14 で +14 cells 改善したが v18_m03 は 78.33% のまま (oversample 希釈)。
    review-only fine-tune で v18_m03 含む難動画を直接学習して打破。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2
import numpy as np

from src.board import HIDDEN_ROWS
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION

LABEL_TO_CODE = {
    "EM": 0, "RED": 1, "BLUE": 2, "GRN": 3,
    "YEL": 4, "PUR": 5, "OJM": 9,
}
PATCH_SIZE = 16


def collect_review(csv_path: Path, video_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists() or not video_path.exists():
        return (np.empty((0, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int32))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (np.empty((0, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int32))
    patches: list[np.ndarray] = []
    labels: list[int] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ans = (row.get("your_answer") or "").strip()
            if not ans or ans not in LABEL_TO_CODE:
                continue
            t = float(row["time"])
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            if fr.shape[:2] != (1080, 1920):
                fr = cv2.resize(fr, (1920, 1080))
            region = (
                DEFAULT_P1_REGION if row["side"] == "1P"
                else DEFAULT_P2_REGION
            )
            r = int(row["row"]) + HIDDEN_ROWS
            c = int(row["col"])
            x1, y1, x2, y2 = region.cell_sample_rect(r, c)
            patch = fr[max(0, y1):min(1080, y2),
                       max(0, x1):min(1920, x2)]
            if patch.size == 0:
                continue
            resized = cv2.resize(
                patch, (PATCH_SIZE, PATCH_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            patches.append(resized)
            labels.append(LABEL_TO_CODE[ans])
    cap.release()
    if not patches:
        return (np.empty((0, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int32))
    return np.stack(patches), np.array(labels, dtype=np.int32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init-model", default="models/cnn_phase_u_v14.pt",
    )
    parser.add_argument(
        "--out-model", default="models/cnn_phase_u_v15.pt",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    base = Path("data/verify/phase_w_review")
    review_specs = [
        ("v05_m55_full", "video_05"),
        ("v12_m54_full", "video_12"),
        ("v09_m02_full", "video_09"),
        ("v13_m02_full", "video_13"),
        ("v17_m11_full", "video_17"),
        ("v18_m03_full", "video_18"),
        ("v18_m08_full", "video_18"),
        ("v18_m15_full", "video_18"),
        ("v19_m06_full", "video_19"),
    ]
    for n in range(4, 20):
        review_specs.append(
            (f"violations_50_bg/v{n:02d}", f"video_{n:02d}")
        )

    all_p: list[np.ndarray] = []
    all_l: list[int] = []
    for name, vid in review_specs:
        ps, ls = collect_review(
            base / name / "labels.csv",
            Path(f"data/frames/{vid}.mp4"),
        )
        if ps.size:
            print(f"  {name}: {ps.shape}")
            all_p.append(ps)
            all_l.extend(ls.tolist())
    patches = np.concatenate(all_p, axis=0)
    labels = np.array(all_l, dtype=np.int32)
    print(f"total review: {patches.shape}")

    # 訓練
    import torch
    from src.patch_classifier import (
        CLASS_INDEX_TO_COLOR, COLOR_TO_CLASS_INDEX, CnnPatchClassifier,
        NUM_CLASSES, PatchSample,
    )

    cls = CnnPatchClassifier()
    state = torch.load(
        args.init_model, map_location="cpu", weights_only=True,
    )
    cls._model.load_state_dict(state)
    print(f"loaded init: {args.init_model}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls._model.to(device)
    cls._model.train()

    labels_idx = np.array([
        COLOR_TO_CLASS_INDEX[int(c)] for c in labels
    ], dtype=np.int64)

    # クラス重み (逆頻度)
    counts = np.bincount(labels_idx, minlength=NUM_CLASSES)
    inv = 1.0 / np.clip(counts, 1, None)
    inv = inv / inv.sum() * NUM_CLASSES
    weight_t = torch.tensor(inv, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_t)
    optimizer = torch.optim.Adam(
        cls._model.parameters(), lr=args.lr, weight_decay=1e-5,
    )

    rng = np.random.default_rng(42)
    n = len(patches)
    print(f"fine-tune: epochs={args.epochs}, lr={args.lr}, n={n}")
    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        total_loss = 0.0
        n_batch = 0
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            batch_patches = patches[idx]
            X = []
            for p in batch_patches:
                resized = cv2.resize(
                    p, (8, 8), interpolation=cv2.INTER_AREA,
                )
                hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
                combined = np.concatenate([resized, hsv], axis=2)
                X.append(combined)
            X_arr = np.stack(X).astype(np.float32) / 255.0
            X_t = torch.from_numpy(X_arr).permute(0, 3, 1, 2).to(device)
            y_t = torch.from_numpy(labels_idx[idx]).to(device)
            optimizer.zero_grad()
            logits = cls._model(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batch += 1
        avg = total_loss / max(1, n_batch)
        print(f"  epoch {epoch + 1}/{args.epochs}: loss={avg:.4f}")
    cls._model.eval()
    cls._model.to("cpu")  # save & eval は CPU で

    out = Path(args.out_model)
    cls.save(out)
    print(f"saved: {to_windows_path(out)}")

    # 簡易 in-sample 評価
    correct = 0
    for i in range(n):
        pred = cls.classify(patches[i])
        truth_color = CLASS_INDEX_TO_COLOR[
            int(labels_idx[i])
        ]
        if pred == truth_color:
            correct += 1
    print(f"in-sample acc: {correct}/{n} = {correct/n:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
