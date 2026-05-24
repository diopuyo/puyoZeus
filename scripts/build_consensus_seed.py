"""Cross-CNN ensemble seed selection.

複数 CNN の合議で全員一致した pseudo label のみを「高信頼 seed」として
別 store に保存する。fine-tune のクラス collapse / 偏った noise を緩和。

設計:
    - 複数 base CNN (例: cnn_phase_b_v1, v2, cnn_phase_u_v17b) を load
    - 全 pseudo label の patch を全 CNN で予測
    - 全員一致 (= len(set(preds)) == 1) かつ pseudo label とも一致した
      sample のみを seed として `data/pseudo_labels_consensus/{vid}/cell.jsonl`
      に書き出す
    - 出力 store は通常の LabelStore 互換 (= phase_i_fine_tune --store-root で
      使える)

使い方:
    python scripts/build_consensus_seed.py \
        --models models/cnn_phase_b_v1.pt models/cnn_phase_b_v2.pt \
                 models/cnn_phase_u_v17b.pt \
        --videos v29 v30 v89 \
        --out-root data/pseudo_labels_consensus \
        [--limit-per-video 50000]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.patch_classifier import (
    CLASS_INDEX_TO_COLOR,
    COLOR_TO_CLASS_INDEX,
    CnnPatchClassifier,
)
from src.self_supervised.label_store import LabelStore
from src.self_supervised.pseudo_label import COMPONENT_CELL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", type=Path, required=True,
                    help="CNN model path のリスト (.pt)")
    p.add_argument("--videos", nargs="+", type=str, required=True,
                    help="入力 video_id のリスト")
    p.add_argument("--in-root", type=Path,
                    default=Path("data/pseudo_labels"),
                    help="入力 pseudo store root")
    p.add_argument("--out-root", type=Path,
                    default=Path("data/pseudo_labels_consensus"),
                    help="seed store の出力先 root")
    p.add_argument("--limit-per-video", type=int, default=0,
                    help="動画あたり処理上限 (0=無制限)")
    p.add_argument("--batch", type=int, default=128,
                    help="CNN inference バッチサイズ")
    return p.parse_args()


def _load_models(paths: list[Path]) -> list[CnnPatchClassifier]:
    """全 CNN を load して GPU 化."""
    models = []
    for path in paths:
        cnn = CnnPatchClassifier()
        st = torch.load(str(path), map_location="cpu", weights_only=True)
        cnn._model.load_state_dict(st)
        cnn._model.eval()
        try:
            if torch.cuda.is_available():
                cnn.to_device("cuda")
        except Exception:
            pass
        models.append(cnn)
    return models


def _batch_predict(
    cnn: CnnPatchClassifier,
    patches: list[np.ndarray],
) -> list[int]:
    """1 CNN で batch 推論し class index list を返す."""
    out: list[int] = []
    for i in range(0, len(patches), 128):
        bp = patches[i:i + 128]
        tensors = [cnn._patch_to_tensor(p)[0] for p in bp]
        batch = cnn._torch.stack(tensors).to(cnn._device)
        with torch.no_grad():
            logits = cnn._model(batch)
        preds = logits.argmax(dim=1).cpu().tolist()
        out.extend(int(x) for x in preds)
    return out


def process_video(
    video_id: str,
    in_root: Path,
    out_root: Path,
    models: list[CnnPatchClassifier],
    limit: int,
) -> dict:
    """1 動画を処理し consensus seed を out_root に書き出す."""
    in_store = LabelStore(video_id=video_id, root=in_root)
    out_store = LabelStore(video_id=video_id, root=out_root)
    samples_iter = in_store.load(COMPONENT_CELL)
    n_in = 0
    n_out = 0
    by_label: dict[int, int] = {}
    buf_samples: list = []
    buf_patches: list[np.ndarray] = []
    BATCH = 256

    def flush() -> None:
        nonlocal n_out
        if not buf_samples:
            return
        # 全 model で予測
        all_preds = [_batch_predict(m, buf_patches) for m in models]
        keep_samples = []
        for i, sample in enumerate(buf_samples):
            try:
                color = int(sample.label)
            except (TypeError, ValueError):
                continue
            true_idx = COLOR_TO_CLASS_INDEX.get(color)
            if true_idx is None:
                continue
            preds_for_i = {p[i] for p in all_preds}
            if len(preds_for_i) == 1 and next(iter(preds_for_i)) == true_idx:
                keep_samples.append(sample)
                by_label[true_idx] = by_label.get(true_idx, 0) + 1
        if keep_samples:
            out_store.append(keep_samples)
            n_out += len(keep_samples)
        buf_samples.clear()
        buf_patches.clear()

    for s in samples_iter:
        if s.component != COMPONENT_CELL:
            continue
        if not isinstance(s.input_data, dict):
            continue
        patch = s.input_data.get("patch")
        if not isinstance(patch, np.ndarray):
            continue
        buf_samples.append(s)
        buf_patches.append(patch)
        n_in += 1
        if len(buf_samples) >= BATCH:
            flush()
        if limit and n_in >= limit:
            break
    flush()
    return {
        "video_id": video_id,
        "n_in": n_in, "n_out": n_out,
        "ratio": n_out / max(1, n_in),
        "by_label": by_label,
    }


def main() -> None:
    args = parse_args()
    print(f"[seed] models={args.models} videos={args.videos}")
    models = _load_models(args.models)
    print(f"[seed] loaded {len(models)} models")
    args.out_root.mkdir(parents=True, exist_ok=True)
    overall_in = overall_out = 0
    t0 = time.time()
    for vid in args.videos:
        t1 = time.time()
        stats = process_video(
            vid, args.in_root, args.out_root, models,
            args.limit_per_video,
        )
        overall_in += stats["n_in"]
        overall_out += stats["n_out"]
        print(
            f"[seed] {vid}: in={stats['n_in']} out={stats['n_out']} "
            f"ratio={stats['ratio']:.3f} by_label={stats['by_label']} "
            f"t={time.time()-t1:.1f}s",
        )
    print(
        f"[seed] DONE total in={overall_in} out={overall_out} "
        f"ratio={overall_out/max(1,overall_in):.3f} "
        f"elapsed={time.time()-t0:.1f}s",
    )


if __name__ == "__main__":
    main()
