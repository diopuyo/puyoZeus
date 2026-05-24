"""Phase J/M の最終 E2E 検証: 重み戦略別の test_acc 比較。

CSV `match_features_v3.csv` (1390 サンプル) を video holdout (video_03 を test)
で評価。Phase J 重み戦略 (global / phase-aware) と既存戦略を比較。

指標:
    - 各 weight_mode (default / phase_j_global / phase_j_aware / etc) の test_acc
    - phase 別の test_acc breakdown
    - 統合精度試算 (score OCR + 視覚版 + 凝視ベース重み)

出力: data/verify/e2e_phase_j_eval.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

CSV_PATH = Path("data/training/match_features_v3.csv")
OUT_JSON = Path("data/verify/e2e_phase_j_eval.json")
TEST_VIDEO = "03"

PHASE_START_LIST = ("start_plus_0", "start_plus_15", "start_plus_30")
PHASE_MID_LIST = ("mid_minus_30", "mid_minus_15", "midpoint",
                   "mid_plus_15", "mid_plus_30")
PHASE_END_LIST = ("end_minus_15", "end_minus_5")


def phase_of(time_phase: str) -> str:
    if time_phase in PHASE_START_LIST:
        return "start"
    if time_phase in PHASE_END_LIST:
        return "end"
    return "mid"


def load() -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    feat = [c for c in rows[0]
            if c not in ("video_id", "match_idx", "time_phase", "label")]
    return rows, feat


def predict_with_weights(rows: list[dict], feat: list[str],
                          weights: dict[str, float]) -> list[int]:
    """各サンプルで sum(features * weights) > 0 なら +1 (1P 勝ち)、else -1。
    incoming_ojama_pressure / opponent_chain_threat 等は重み中の符号で吸収。
    """
    preds: list[int] = []
    for r in rows:
        score = 0.0
        for c in feat:
            score += float(r[c]) * weights.get(c, 0.0)
        preds.append(1 if score > 0 else -1)
    return preds


def predict_with_phase_aware(
    rows: list[dict], feat: list[str],
    weights_by_phase: dict[str, dict[str, float]],
) -> list[int]:
    preds: list[int] = []
    for r in rows:
        ph = phase_of(r["time_phase"])
        weights = weights_by_phase[ph]
        score = sum(float(r[c]) * weights.get(c, 0.0) for c in feat)
        preds.append(1 if score > 0 else -1)
    return preds


def evaluate(rows: list[dict], preds: list[int]) -> dict:
    truths = [int(r["label"]) for r in rows]
    test_idx = [i for i, r in enumerate(rows) if r["video_id"] == TEST_VIDEO]
    train_idx = [i for i, r in enumerate(rows) if r["video_id"] != TEST_VIDEO]

    def acc(idxs: list[int]) -> float:
        if not idxs:
            return 0.0
        correct = sum(1 for i in idxs if preds[i] == truths[i])
        return correct / len(idxs)

    # phase 別
    by_phase: dict[str, list[int]] = defaultdict(list)
    for i in test_idx:
        by_phase[phase_of(rows[i]["time_phase"])].append(i)
    phase_acc = {p: acc(idxs) for p, idxs in by_phase.items()}

    return {
        "train_acc": acc(train_idx),
        "test_acc": acc(test_idx),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "phase_acc": phase_acc,
    }


def main() -> int:
    from src.scorer import (
        DEFAULT_WEIGHTS,
        LEARNED_WEIGHTS_GLOBAL,
        LEARNED_WEIGHTS_V3_GLOBAL,
        LEARNED_WEIGHTS_PHASE_J_GLOBAL,
        OPTIMAL_PHASE_WEIGHTS,
        PHASE_J_PHASE_WEIGHTS,
        PHASE_WEIGHT_MAP,
    )

    rows, feat = load()
    print(f"loaded {len(rows)} samples, {len(feat)} features")

    strategies: list[tuple[str, callable]] = [
        ("DEFAULT", lambda: predict_with_weights(rows, feat, DEFAULT_WEIGHTS)),
        ("LEARNED_GLOBAL",
         lambda: predict_with_weights(rows, feat, LEARNED_WEIGHTS_GLOBAL)),
        ("LEARNED_V3_GLOBAL",
         lambda: predict_with_weights(rows, feat, LEARNED_WEIGHTS_V3_GLOBAL)),
        ("LEARNED_PHASE_J_GLOBAL",
         lambda: predict_with_weights(rows, feat,
                                      LEARNED_WEIGHTS_PHASE_J_GLOBAL)),
        ("PhaseAware_learned",
         lambda: predict_with_phase_aware(rows, feat, PHASE_WEIGHT_MAP)),
        ("PhaseAware_optimal",
         lambda: predict_with_phase_aware(rows, feat, OPTIMAL_PHASE_WEIGHTS)),
        ("PhaseAware_phase_j",
         lambda: predict_with_phase_aware(rows, feat, PHASE_J_PHASE_WEIGHTS)),
    ]

    out: dict = {}
    print(f"\n{'strategy':<28s} test_acc  start  mid    end")
    for name, fn in strategies:
        preds = fn()
        ev = evaluate(rows, preds)
        out[name] = ev
        ph = ev["phase_acc"]
        print(f"  {name:<28s} {ev['test_acc']:.3f}     "
              f"{ph.get('start',0):.3f}  {ph.get('mid',0):.3f}  "
              f"{ph.get('end',0):.3f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"\n[save] {OUT_JSON}")

    # ベスト戦略
    best = max(strategies, key=lambda x: out[x[0]]["test_acc"])
    best_name, _ = best
    print(f"\n[best strategy] {best_name}: "
          f"test_acc={out[best_name]['test_acc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
