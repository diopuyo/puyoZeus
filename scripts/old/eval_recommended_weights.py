"""
段階 4-4: RECOMMENDED 重みを含む 4 戦略の精度比較

目的:
    DEFAULT / LEARNED_GLOBAL / LEARNED_V3_GLOBAL / RECOMMENDED の 4 戦略を
    1390 サンプル video_holdout test (n=390) と各 time_phase で評価し、
    accuracy 比較表を出力する。

入力:
    data/training/match_features_v2.csv

出力:
    data/verify/eval_recommended_weights.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.old.eda_features import Dataset, load_dataset  # noqa: E402
from src.old.scorer import (  # noqa: E402
    DEFAULT_WEIGHTS,
    LEARNED_WEIGHTS_GLOBAL,
    LEARNED_WEIGHTS_RECOMMENDED,
    LEARNED_WEIGHTS_V3_GLOBAL,
)

# ============================
# 定数
# ============================

DEFAULT_INPUT_CSV: Path = Path("data/training/match_features_v2.csv")
DEFAULT_OUTPUT_JSON: Path = Path("data/verify/eval_recommended_weights.json")
TEST_VIDEO: str = "03"

STRATEGY_WEIGHTS: dict[str, dict[str, float]] = {
    "DEFAULT": DEFAULT_WEIGHTS,
    "LEARNED_GLOBAL": LEARNED_WEIGHTS_GLOBAL,
    "LEARNED_V3_GLOBAL": LEARNED_WEIGHTS_V3_GLOBAL,
    "RECOMMENDED": LEARNED_WEIGHTS_RECOMMENDED,
}


# ============================
# 評価
# ============================


def evaluate_strategy(
    ds: Dataset, weights: dict[str, float], mask: np.ndarray,
) -> float:
    """指定マスクのサンプルで重み付けスコアの符号予測精度を返す。"""
    if mask.sum() == 0:
        return 0.0
    w = np.array([weights.get(n, 0.0) for n in ds.feature_names])
    z = ds.X[mask] @ w
    pred = np.where(z >= 0, 1, -1)
    return float((pred == ds.y[mask]).mean())


def evaluate_all_phases(
    ds: Dataset, test_mask: np.ndarray,
) -> dict[str, dict[str, dict[str, float]]]:
    """全戦略 × 全 phase の精度を計算する。"""
    phases = sorted(set(ds.time_phases))
    out: dict[str, dict[str, dict[str, float]]] = {
        name: {} for name in STRATEGY_WEIGHTS
    }
    for phase in ["__overall__"] + phases:
        if phase == "__overall__":
            m = test_mask
        else:
            phase_mask = np.array([p == phase for p in ds.time_phases])
            m = test_mask & phase_mask
        for name, w in STRATEGY_WEIGHTS.items():
            out[name][phase] = {
                "n": int(m.sum()),
                "test_acc": evaluate_strategy(ds, w, m),
            }
    return out


def best_per_phase(
    accs: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, Any]]:
    """phase ごとに最良戦略を求める。"""
    out: dict[str, dict[str, Any]] = {}
    phases = list(next(iter(accs.values())).keys())
    for phase in phases:
        best = max(
            accs.keys(), key=lambda s: accs[s][phase]["test_acc"],
        )
        out[phase] = {
            "best_strategy": best,
            "test_acc": accs[best][phase]["test_acc"],
            "n": accs[best][phase]["n"],
        }
    return out


# ============================
# main
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(description="RECOMMENDED 評価")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--test-video", type=str, default=TEST_VIDEO)
    args = parser.parse_args()

    ds = load_dataset(args.csv)
    test_mask = np.array([v == args.test_video for v in ds.video_ids])
    print(f"[load] n={len(ds.y)}, holdout n={int(test_mask.sum())}")

    accs = evaluate_all_phases(ds, test_mask)
    bests = best_per_phase(accs)

    print("\n=== Strategy comparison (test_acc on video_holdout) ===")
    print(f"{'phase':<20} {'DEFAULT':>10} {'GLOBAL':>10} "
          f"{'V3':>10} {'REC':>10} {'best':>14}")
    phases_ordered = ["__overall__"] + sorted(
        set(ds.time_phases),
    )
    for phase in phases_ordered:
        row = [accs[s][phase]["test_acc"] for s in STRATEGY_WEIGHTS]
        best = bests[phase]["best_strategy"]
        print(
            f"{phase:<20} " + " ".join(f"{v:>10.4f}" for v in row) +
            f"   {best:>10}",
        )

    out: dict[str, Any] = {
        "csv_path": str(args.csv),
        "n_samples": int(len(ds.y)),
        "test_video": args.test_video,
        "n_holdout": int(test_mask.sum()),
        "strategies": list(STRATEGY_WEIGHTS.keys()),
        "accuracy_table": accs,
        "best_per_phase": bests,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
