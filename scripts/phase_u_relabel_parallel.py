"""Phase U V1.3': parallel/ 弱ラベル npz を CNN v6 で再ラベル + 高信頼度フィルタ。

`data/training/parallel/*.npz` の各ファイルを 1 つずつロードし、
CNN v6 で predict_proba → max prob >= threshold のパッチのみ採用。
新ラベル (argmax) を付けて `data/training_phase_u/parallel_relabeled/{tag}.npz`
へ書き出す (個別保存で WSL2 OOM 回避)。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_u_relabel_parallel
"""
from __future__ import annotations

import argparse
import os
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

from src.patch_classifier import (
    CLASS_INDEX_TO_COLOR,
    CnnPatchClassifier,
    PATCH_RESIZE_H,
    PATCH_RESIZE_W,
)


def load_cnn(model_path: str, device: str) -> CnnPatchClassifier:
    """CNN v6 をロードして指定デバイスへ移す。"""
    cnn = CnnPatchClassifier()
    state = torch.load(model_path, map_location=device, weights_only=True)
    cnn._model.load_state_dict(state)
    cnn._model.to(device)
    cnn._model.eval()
    return cnn


def patches_to_tensor(patches: np.ndarray, device: str) -> torch.Tensor:
    """N x H x W x 3 BGR uint8 -> N x 6 x 8 x 8 normalized tensor.

    cv2.resize と HSV 変換は CPU 側でバッチ処理。
    """
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
    # NHWC -> NCHW
    tensor = torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()
    return tensor.to(device)


def relabel_one(
    npz_path: Path,
    cnn: CnnPatchClassifier,
    device: str,
    threshold: float,
    batch: int,
    require_hsv_match: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """1 npz を再ラベル + 閾値フィルタ。(accepted_patches, accepted_labels, in, out).

    Args:
        require_hsv_match: True なら HSV ラベル (元 npz) と CNN v6 ラベル
            両方一致時のみ採用 (誤認確率を下げる強フィルタ)。
    """
    d = np.load(npz_path)
    patches = d["patches"]
    hsv_labels = d["labels"].astype(np.int64)
    N = patches.shape[0]

    max_probs = np.zeros(N, dtype=np.float32)
    new_labels = np.zeros(N, dtype=np.int64)

    for s in range(0, N, batch):
        e = min(s + batch, N)
        tensor = patches_to_tensor(patches[s:e], device)
        with torch.no_grad():
            logits = cnn._model(tensor)
            probs = torch.softmax(logits, dim=1)
            max_p, idx = probs.max(dim=1)
        max_probs[s:e] = max_p.cpu().numpy()
        idx_np = idx.cpu().numpy()
        for j, ci in enumerate(idx_np):
            new_labels[s + j] = CLASS_INDEX_TO_COLOR[ci]

    mask = max_probs >= threshold
    if require_hsv_match:
        mask &= (new_labels == hsv_labels)
    return patches[mask], new_labels[mask], N, int(mask.sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cnn-model", default="models/cnn_phase_u_v6.pt",
    )
    parser.add_argument(
        "--input-dir", default="data/training/parallel",
    )
    parser.add_argument(
        "--out-dir",
        default="data/training_phase_u/parallel_relabeled",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.90,
        help="max prob 閾値 (これ以上のみ採用)",
    )
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument(
        "--device", default="auto",
        help="cuda / cpu / auto",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="処理する npz 数の上限 (0 = 無制限、デバッグ用)",
    )
    parser.add_argument(
        "--require-hsv-match", action="store_true",
        help="HSV (元 npz) と CNN v6 ラベルが一致したパッチのみ採用",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"device: {device}")
    print(f"cnn:    {args.cnn_model}")
    print(f"thresh: {args.threshold}")

    cnn = load_cnn(args.cnn_model, device=device)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(in_dir.glob("*.npz"))
    if args.limit > 0:
        npz_files = npz_files[: args.limit]
    print(f"input: {len(npz_files)} npz files")

    total_in = 0
    total_out = 0
    summary_rows: list[str] = []
    summary_rows.append("file\tin\tout\tratio")

    for i, npz_path in enumerate(npz_files, 1):
        out_path = out_dir / npz_path.name
        if out_path.exists():
            # 再開対応: 既処理スキップ
            d = np.load(out_path)
            kept = d["patches"].shape[0]
            d2 = np.load(npz_path)
            n_in = d2["patches"].shape[0]
            total_in += n_in
            total_out += kept
            print(
                f"[{i}/{len(npz_files)}] {npz_path.name}: "
                f"SKIP (already done), {kept}/{n_in}"
            )
            summary_rows.append(
                f"{npz_path.name}\t{n_in}\t{kept}\t{kept / max(1, n_in):.3f}"
            )
            continue

        try:
            kp, kl, n_in, n_out = relabel_one(
                npz_path, cnn, device, args.threshold, args.batch,
                require_hsv_match=args.require_hsv_match,
            )
        except Exception as e:
            print(f"[{i}] {npz_path.name}: ERROR {e}")
            continue

        np.savez_compressed(out_path, patches=kp, labels=kl)
        total_in += n_in
        total_out += n_out
        ratio = n_out / max(1, n_in)
        print(
            f"[{i}/{len(npz_files)}] {npz_path.name}: "
            f"{n_out}/{n_in} ({100 * ratio:.1f}%), "
            f"total {total_out}/{total_in}"
        )
        summary_rows.append(
            f"{npz_path.name}\t{n_in}\t{n_out}\t{ratio:.3f}"
        )

    summary_path = out_dir / "summary.tsv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_rows) + "\n")
    print(f"\nsummary: {to_windows_path(summary_path)}")
    print(f"final:  {total_out}/{total_in} accepted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
