"""
TTA の即効性検証: 既存 cnn_global_best.pt を holdout で評価し
通常推論 vs TTA 推論の精度を比較する。

使い方:
    ./venv/bin/python scripts/eval_tta.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from src.patch_classifier import CnnPatchClassifier
from src.tta import TTAClassifier
from scripts.e2e_validate import _pick_holdout_npz

NAME_MAP = {0: "空", 1: "赤", 2: "青", 3: "緑", 4: "黄", 5: "紫", 9: "お邪魔"}


def evaluate(cnn_or_tta, patches, labels, indices, label_prefix: str) -> dict:
    correct = 0
    per_class_correct: dict[int, int] = {}
    per_class_total: dict[int, int] = {}
    t0 = time.time()
    for i in indices:
        pred = cnn_or_tta.classify(patches[i])
        true = int(labels[i])
        per_class_total[true] = per_class_total.get(true, 0) + 1
        if pred == true:
            correct += 1
            per_class_correct[true] = per_class_correct.get(true, 0) + 1
    elapsed = time.time() - t0
    n = len(indices)
    acc = correct / n
    print(f"\n{label_prefix} 全体精度: {acc:.4f} ({correct}/{n}) 推論時間 {elapsed:.1f}s")
    for code, total in sorted(per_class_total.items()):
        c = per_class_correct.get(code, 0)
        a = c / total if total > 0 else 0
        print(f"  {NAME_MAP.get(code, code)} (n={total}): {a:.4f}")
    return {
        "acc": acc,
        "elapsed_sec": elapsed,
        "per_class_total": per_class_total,
        "per_class_correct": per_class_correct,
    }


def main() -> int:
    cnn_path = Path("models/cnn_global_best.pt")
    if not cnn_path.exists():
        print(f"モデルなし: {cnn_path}", file=sys.stderr)
        return 1
    holdout = _pick_holdout_npz()
    if not holdout:
        print("ホールドアウト npz が見つからない", file=sys.stderr)
        return 1
    print(f"CNN: {cnn_path}")
    print(f"Holdout: {holdout}")

    data = np.load(holdout)
    patches = data["patches"]
    labels = data["labels"]
    rng = np.random.default_rng(0)
    n = min(2000, len(labels))   # TTA は 5x なので 5000 → 2000 に減
    idx = rng.choice(len(labels), n, replace=False)

    cnn = CnnPatchClassifier.load(cnn_path)
    base_result = evaluate(cnn, patches, labels, idx, "[BASE]")

    tta = TTAClassifier(cnn)
    tta_result = evaluate(tta, patches, labels, idx, "[TTA ]")

    diff = (tta_result["acc"] - base_result["acc"]) * 100
    print(f"\n=== 結果 ===")
    print(f"BASE: acc={base_result['acc']:.4f}, time={base_result['elapsed_sec']:.1f}s")
    print(f"TTA : acc={tta_result['acc']:.4f}, time={tta_result['elapsed_sec']:.1f}s")
    print(f"差分: {diff:+.2f}pt  (TTA は {tta_result['elapsed_sec']/max(0.001, base_result['elapsed_sec']):.1f}x 遅い)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
