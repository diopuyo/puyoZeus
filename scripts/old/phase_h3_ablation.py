"""Phase H3 弱指標削除 + Tier 別 ablation スクリプト.

目的:
    Phase H2 の 280 特徴量の中から弱指標を段階的に削除し、最適な特徴部分集合を
    発見する。Permutation importance を ranking 基準として feature を 5 tier に
    分類し、累積構成 (Tier S only / S+A / S+A+B / S+A+B+C / 全部) で
    LR + sklearn HGBT の video holdout test acc + LOOV phase mean を比較する。

入力:
    --csv data/training/match_features_phase_h2_quick_phased.csv
出力:
    --out data/verify/phase_h3_ablation_results.json (tier 別効果表 + 推奨 subset)

注意:
    - lightgbm は WSL の libgomp 未導入で import 不可のため、H2 と同じ
      sklearn `HistGradientBoostingClassifier` を GBM 代替として使用する。
    - feature 自動検出 (META_COLS 以外を全て feature とみなす)。
    - 1 関数 50 行以内、マジックナンバー禁止。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ============================
# 定数 (マジックナンバー回避)
# ============================
META_COLS = {"video_id", "match_idx", "time_phase", "frame_idx", "timestamp", "label"}
PHASE_GROUPS: dict[str, tuple[str, ...]] = {
    "start": ("start_plus_20",),
    "mid": ("mid_minus_20", "midpoint", "mid_plus_20"),
    "end": ("end_minus_5",),
}

# Tier 境界 (累積上限 index、Permutation rank の昇順、つまり大きいほど弱い)
TIER_S_END = 20      # rank 0..19   = Tier S (top 20、必須)
TIER_A_END = 50      # rank 20..49  = Tier A (中)
TIER_B_END = 100     # rank 50..99  = Tier B (中-弱)
TIER_C_END = 200     # rank 100..199 = Tier C (弱)
# rank 200..   = Tier D (極弱、削除候補)

# 学習パラメータ
HGBT_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "max_iter": 300,
    "min_samples_leaf": 20,
    "max_leaf_nodes": 31,
}
LR_C = 0.5  # H2 best
N_TEST_VIDEOS = 3
RANDOM_SEED = 0
PERM_REPEATS = 5


def load_h2_csv(path: Path) -> dict[str, Any]:
    """H2 csv を読み込み、自動で feature columns を抽出する."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    feat_cols = [c for c in fieldnames if c not in META_COLS]
    n, d = len(rows), len(feat_cols)
    X = np.zeros((n, d), dtype=np.float32)
    y = np.zeros(n, dtype=np.int8)
    video_ids: list[str] = []
    time_phases: list[str] = []
    for i, r in enumerate(rows):
        for j, c in enumerate(feat_cols):
            X[i, j] = float(r.get(c, 0.0) or 0.0)
        y[i] = int(r["label"])
        video_ids.append(r["video_id"])
        time_phases.append(r.get("time_phase", "midpoint"))
    return {
        "X": X, "y": y,
        "video_ids": np.array(video_ids),
        "time_phases": np.array(time_phases),
        "feat_cols": feat_cols,
        "n": n, "d": d,
    }


def video_holdout_split(
    video_ids: np.ndarray, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """動画単位で test ホールドアウトを作成する."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(video_ids)
    if len(uniq) <= n_test:
        n_test = max(1, len(uniq) // 3)
    test_videos = rng.choice(uniq, size=n_test, replace=False)
    test_mask = np.isin(video_ids, test_videos)
    return ~test_mask, test_mask


def fit_hgbt(X_tr, y_tr, X_te, y_te) -> tuple[float, float, Any]:
    """sklearn HistGradientBoosting で fit / score を返す."""
    y_tr_b = (y_tr > 0).astype(int)
    y_te_b = (y_te > 0).astype(int)
    clf = HistGradientBoostingClassifier(random_state=RANDOM_SEED, **HGBT_PARAMS)
    clf.fit(X_tr, y_tr_b)
    train = float(clf.score(X_tr, y_tr_b))
    test = float(clf.score(X_te, y_te_b)) if len(X_te) > 0 else 0.0
    return train, test, clf


def fit_lr(X_tr, y_tr, X_te, y_te) -> tuple[float, float, Any]:
    """L2 LogisticRegression で fit / score を返す."""
    y_tr_b = (y_tr > 0).astype(int)
    y_te_b = (y_te > 0).astype(int)
    clf = LogisticRegression(C=LR_C, penalty="l2", max_iter=2000, random_state=RANDOM_SEED)
    clf.fit(X_tr, y_tr_b)
    train = float(clf.score(X_tr, y_tr_b))
    test = float(clf.score(X_te, y_te_b)) if len(X_te) > 0 else 0.0
    return train, test, clf


def loov_evaluate(ds: dict, fit_fn) -> tuple[float, float]:
    """Leave-One-Out-on-Video で平均 acc / std を返す."""
    uniq = np.unique(ds["video_ids"])
    accs = []
    for vid in uniq:
        test_mask = ds["video_ids"] == vid
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        try:
            _, te, _ = fit_fn(
                ds["X"][train_mask], ds["y"][train_mask],
                ds["X"][test_mask], ds["y"][test_mask],
            )
            accs.append(te)
        except Exception as e:
            print(f"  LOOV vid={vid} skip: {e}")
    if not accs:
        return 0.0, 0.0
    return float(np.mean(accs)), float(np.std(accs))


def loov_phase_mean(ds: dict, fit_fn) -> dict[str, tuple[float, float]]:
    """time_phase グループ別の LOOV を返す."""
    out: dict[str, tuple[float, float]] = {}
    for phase_name, phases in PHASE_GROUPS.items():
        phase_mask = np.isin(ds["time_phases"], phases)
        if phase_mask.sum() == 0:
            out[phase_name] = (0.0, 0.0)
            continue
        sub = {
            "X": ds["X"][phase_mask],
            "y": ds["y"][phase_mask],
            "video_ids": ds["video_ids"][phase_mask],
            "time_phases": ds["time_phases"][phase_mask],
        }
        out[phase_name] = loov_evaluate(sub, fit_fn)
    return out


def compute_permutation_ranking(
    ds: dict, train_mask: np.ndarray, test_mask: np.ndarray
) -> list[int]:
    """全 feature で baseline HGBT を学習し、permutation importance で rank index を返す.

    返り値: feature index の list、importance の降順 (0 番目 = 最強).
    """
    print("[rank] baseline HGBT で全 feature を学習...")
    _, _, clf = fit_hgbt(
        ds["X"][train_mask], ds["y"][train_mask],
        ds["X"][test_mask], ds["y"][test_mask],
    )
    y_te_b = (ds["y"][test_mask] > 0).astype(int)
    perm = permutation_importance(
        clf, ds["X"][test_mask], y_te_b,
        n_repeats=PERM_REPEATS, random_state=RANDOM_SEED, n_jobs=4,
    )
    order = np.argsort(perm.importances_mean)[::-1]
    print(f"[rank] permutation importance computed, total features={len(order)}")
    return [int(i) for i in order]


def assign_tiers(ranking: list[int], total: int) -> dict[str, list[int]]:
    """ranking (importance 降順) を 5 tier に分類する."""
    tiers = {
        "S": ranking[:TIER_S_END],
        "A": ranking[TIER_S_END:TIER_A_END],
        "B": ranking[TIER_A_END:TIER_B_END],
        "C": ranking[TIER_B_END:TIER_C_END],
        "D": ranking[TIER_C_END:total],
    }
    return tiers


def build_cumulative_subsets(tiers: dict[str, list[int]]) -> dict[str, list[int]]:
    """累積 tier 構成 (S only / S+A / ... / 全部) を返す."""
    subsets: dict[str, list[int]] = {}
    cur: list[int] = []
    for name in ("S", "A", "B", "C", "D"):
        cur = cur + list(tiers[name])
        key = "+".join(["S", "A", "B", "C", "D"][: ["S", "A", "B", "C", "D"].index(name) + 1])
        subsets[key] = list(cur)
    return subsets


def evaluate_subset(ds: dict, indices: list[int]) -> dict[str, Any]:
    """与えられた feature index 部分集合で video holdout + LOOV phase 平均を測定."""
    sub_ds = {
        "X": ds["X"][:, indices],
        "y": ds["y"],
        "video_ids": ds["video_ids"],
        "time_phases": ds["time_phases"],
        "feat_cols": [ds["feat_cols"][i] for i in indices],
    }
    train_mask, test_mask = video_holdout_split(
        sub_ds["video_ids"], n_test=N_TEST_VIDEOS, seed=RANDOM_SEED
    )
    out: dict[str, Any] = {"n_features": len(indices)}

    _, lr_te, _ = fit_lr(
        sub_ds["X"][train_mask], sub_ds["y"][train_mask],
        sub_ds["X"][test_mask], sub_ds["y"][test_mask],
    )
    _, hgbt_te, _ = fit_hgbt(
        sub_ds["X"][train_mask], sub_ds["y"][train_mask],
        sub_ds["X"][test_mask], sub_ds["y"][test_mask],
    )
    out["lr_video_holdout"] = lr_te
    out["hgbt_video_holdout"] = hgbt_te

    lr_phase = loov_phase_mean(sub_ds, fit_lr)
    hgbt_phase = loov_phase_mean(sub_ds, fit_hgbt)
    out["lr_phase_loov"] = {p: {"mean": m, "std": s} for p, (m, s) in lr_phase.items()}
    out["hgbt_phase_loov"] = {p: {"mean": m, "std": s} for p, (m, s) in hgbt_phase.items()}
    out["lr_phase_avg"] = float(np.mean([m for m, _ in lr_phase.values()]))
    out["hgbt_phase_avg"] = float(np.mean([m for m, _ in hgbt_phase.values()]))
    return out


def select_recommended_subset(
    results: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """tier 累積構成の中で best score を選ぶ.

    score = LR video holdout + LR phase avg (重視: video holdout 厳密性).
    """
    best_key, best_score, best_val = "", -1.0, {}
    for key, val in results.items():
        score = float(val.get("lr_video_holdout", 0.0)) + float(val.get("lr_phase_avg", 0.0))
        if score > best_score:
            best_score, best_key, best_val = score, key, val
    return best_key, best_val


def print_summary(results: dict[str, dict[str, Any]]) -> None:
    """tier 別 ablation 結果を表形式で標準出力."""
    print("\n=== Phase H3 Ablation Summary ===")
    header = (
        f"{'subset':<14} {'#feat':>6} "
        f"{'LR_VH':>7} {'LR_avg':>7} {'HGBT_VH':>8} {'HGBT_avg':>8}"
    )
    print(header)
    print("-" * len(header))
    for key, val in results.items():
        print(
            f"{key:<14} {val['n_features']:>6} "
            f"{val['lr_video_holdout']:>7.3f} {val['lr_phase_avg']:>7.3f} "
            f"{val['hgbt_video_holdout']:>8.3f} {val['hgbt_phase_avg']:>8.3f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ds = load_h2_csv(args.csv)
    print(f"[load] n={ds['n']}, d={ds['d']}, videos={len(np.unique(ds['video_ids']))}")

    train_mask, test_mask = video_holdout_split(
        ds["video_ids"], n_test=N_TEST_VIDEOS, seed=RANDOM_SEED
    )
    ranking = compute_permutation_ranking(ds, train_mask, test_mask)
    tiers = assign_tiers(ranking, ds["d"])
    print("[tier] sizes:", {k: len(v) for k, v in tiers.items()})

    subsets = build_cumulative_subsets(tiers)
    results: dict[str, dict[str, Any]] = {}
    for name, indices in subsets.items():
        print(f"\n[eval] subset={name}, #feat={len(indices)}")
        results[name] = evaluate_subset(ds, indices)
        r = results[name]
        print(
            f"  LR vh={r['lr_video_holdout']:.3f} avg={r['lr_phase_avg']:.3f} "
            f"| HGBT vh={r['hgbt_video_holdout']:.3f} avg={r['hgbt_phase_avg']:.3f}"
        )

    print_summary(results)
    best_key, best_val = select_recommended_subset(results)
    print(f"\n[recommend] best subset = {best_key} (#feat={best_val['n_features']})")

    # tier 別 feature name dump (削除候補 = Tier D)
    tier_names = {
        tname: [ds["feat_cols"][i] for i in idxs] for tname, idxs in tiers.items()
    }

    out_payload = {
        "n": ds["n"],
        "d": ds["d"],
        "tier_sizes": {k: len(v) for k, v in tiers.items()},
        "tier_features": tier_names,
        "ablation": results,
        "recommended_subset": best_key,
        "recommended_n_features": best_val["n_features"],
        "drop_candidates": tier_names["D"],
        "baseline_h2": {
            "lr_video_holdout": 0.7396,
            "lr_phase_avg": 0.6669,
            "hgbt_video_holdout": 0.6724,
            "hgbt_phase_avg": 0.6412,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
