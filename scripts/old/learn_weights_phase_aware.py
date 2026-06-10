"""Phase J 21 特徴量で phase 別 (start / mid / end) に重み学習する。

PhaseAwareScorer の LEARNED_WEIGHTS_PHASE_J_START/MID/END を生成。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DEFAULT_CSV = Path("data/training/match_features_v3.csv")
OUT_JSON = Path("data/verify/learned_weights_phase_j_phase_aware.json")

# Phase 分類
PHASE_START_LIST = ("start_plus_0", "start_plus_15", "start_plus_30")
PHASE_MID_LIST = (
    "mid_minus_30", "mid_minus_15", "midpoint",
    "mid_plus_15", "mid_plus_30",
)
PHASE_END_LIST = ("end_minus_15", "end_minus_5")
TEST_VIDEO = "03"


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """CSV を読み、X, y, video_ids, time_phases, feature_names を返す。"""
    rows: list[dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    feature_names = [
        c for c in rows[0].keys()
        if c not in ("video_id", "match_idx", "time_phase", "label")
    ]
    X = np.array([[float(r[c]) for c in feature_names] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    video_ids = [r["video_id"] for r in rows]
    time_phases = [r["time_phase"] for r in rows]
    return X, y, video_ids, time_phases, feature_names


def video_holdout_split(
    video_ids: list[str], test_video: str
) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.array([i for i, v in enumerate(video_ids) if v != test_video])
    test_idx = np.array([i for i, v in enumerate(video_ids) if v == test_video])
    return train_idx, test_idx


def filter_phase(
    time_phases: list[str], allowed: tuple[str, ...]
) -> np.ndarray:
    return np.array(
        [i for i, p in enumerate(time_phases) if p in allowed]
    )


def train_phase(
    X: np.ndarray, y: np.ndarray, video_ids: list[str], time_phases: list[str],
    phase_list: tuple[str, ...], phase_name: str, feature_names: list[str],
) -> dict:
    """指定 phase 限定で重み学習し、weights dict を返す。"""
    phase_idx = filter_phase(time_phases, phase_list)
    X_phase = X[phase_idx]
    y_phase = y[phase_idx]
    video_phase = [video_ids[i] for i in phase_idx]
    train_idx, test_idx = video_holdout_split(video_phase, TEST_VIDEO)
    X_tr, y_tr = X_phase[train_idx], y_phase[train_idx]
    X_te, y_te = X_phase[test_idx], y_phase[test_idx]
    print(f"\n=== {phase_name} ({len(phase_list)} phases) ===")
    print(f"  total={len(phase_idx)} train={len(train_idx)} test={len(test_idx)}")

    # 標準化 + L1 LR (sparsity 重視)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    best_acc = 0.0
    best_model = None
    best_C = None
    for C in (0.05, 0.1, 0.5, 1.0, 5.0):
        m = LogisticRegression(
            C=C, penalty="l1", solver="saga", max_iter=2000,
        )
        m.fit(X_tr_s, y_tr)
        acc = m.score(X_te_s, y_te)
        if acc > best_acc:
            best_acc = acc
            best_model = m
            best_C = C
    print(f"  best LR L1 C={best_C}: test_acc={best_acc:.3f}")

    # 重みを原スケールに戻す: w_orig = w_std / std_x
    coef_std = best_model.coef_[0]
    coef_orig = coef_std / np.maximum(scaler.scale_, 1e-9)
    weights = {n: float(coef_orig[i]) for i, n in enumerate(feature_names)}
    return {
        "phase_name": phase_name,
        "phase_list": list(phase_list),
        "test_acc": float(best_acc),
        "best_C": best_C,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "weights": weights,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    X, y, video_ids, time_phases, feature_names = load_dataset(args.csv)
    print(f"loaded: {len(X)} samples, {len(feature_names)} features")
    print(f"label dist: +1={(y==1).sum()} -1={(y==-1).sum()}")

    results = {}
    results["start"] = train_phase(
        X, y, video_ids, time_phases, PHASE_START_LIST, "start", feature_names,
    )
    results["mid"] = train_phase(
        X, y, video_ids, time_phases, PHASE_MID_LIST, "mid", feature_names,
    )
    results["end"] = train_phase(
        X, y, video_ids, time_phases, PHASE_END_LIST, "end", feature_names,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[save] {args.out}")
    print()
    print("=== phase 別 test_acc ===")
    for phase in ("start", "mid", "end"):
        r = results[phase]
        print(f"  {phase:6s}: {r['test_acc']:.3f} (n_test={r['n_test']})")
    avg = np.mean([results[p]["test_acc"] for p in ("start", "mid", "end")])
    print(f"  average: {avg:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
