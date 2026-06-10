"""Phase H3 Mixed-effects model 検証スクリプト.

目的:
    動画別 random intercept を導入し、LOOV variance を補正できるか検証する。
    statsmodels.MixedLM (linear mixed model) を線形回帰として使い、
    動画 ID を group とすることで within-video bias を分離する。
    test 動画は完全に未知のため video holdout 評価では random intercept は
    効かないが、混合モデルで学習した固定効果重みを使った予測精度を測ることで
    「動画別ノイズを除去した上での fixed effect の推定精度」を確認する。

    参考: lightgbm が WSL libgomp 未導入で利用不可なため native categorical 経路は
    取れず、statsmodels.MixedLM を採用する (要件で許容済).

入力:
    --csv data/training/match_features_phase_h2_quick_phased.csv
出力:
    --out data/verify/phase_h3_mixed_effects.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
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
N_TEST_VIDEOS = 3
RANDOM_SEED = 0
LR_C = 0.5
TOP_K_FEATURES = 30  # MixedLM は計算重いので feature 数を絞る
MIXEDLM_MAX_ITER = 200


def load_h2_csv(path: Path) -> dict[str, Any]:
    """H2 csv を読み込む (ablation スクリプトと同じ仕様)."""
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
    """動画単位で test ホールドアウトを作成."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(video_ids)
    if len(uniq) <= n_test:
        n_test = max(1, len(uniq) // 3)
    test_videos = rng.choice(uniq, size=n_test, replace=False)
    test_mask = np.isin(video_ids, test_videos)
    return ~test_mask, test_mask


def select_top_features(X: np.ndarray, y: np.ndarray, k: int) -> list[int]:
    """L2-LR の |coef| top k feature index を返す (MixedLM の前処理).

    MixedLM は計算量重いため、まず全特徴量で LR を学習して |coef| 上位を抽出する。
    """
    y_bin = (y > 0).astype(int)
    clf = LogisticRegression(C=LR_C, penalty="l2", max_iter=2000, random_state=RANDOM_SEED)
    clf.fit(X, y_bin)
    coef = np.abs(clf.coef_.ravel())
    top = np.argsort(coef)[::-1][:k]
    return [int(i) for i in top]


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """列毎に z-score 化する (mean/std を返す)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd, mu, sd


def fit_mixedlm_train(
    X_tr: np.ndarray, y_tr: np.ndarray, groups_tr: np.ndarray
) -> tuple[Any, np.ndarray, float]:
    """statsmodels MixedLM を train セットで学習.

    返り値: (fitted result, fixed effects coef ベクトル, intercept).
    target は連続値とみなし linear mixed model で扱う (二値でも近似的に有効).
    """
    import statsmodels.api as sm

    y_bin = (y_tr > 0).astype(np.float64)
    exog = sm.add_constant(X_tr.astype(np.float64), has_constant="add")
    model = sm.MixedLM(endog=y_bin, exog=exog, groups=groups_tr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(method="lbfgs", maxiter=MIXEDLM_MAX_ITER)
    fe = np.asarray(res.fe_params, dtype=np.float64)
    intercept = float(fe[0])
    coef = fe[1:]
    return res, coef, intercept


def predict_mixed(
    X_te: np.ndarray, coef: np.ndarray, intercept: float
) -> np.ndarray:
    """fixed effects のみで予測 (test の動画は未知 → random intercept は 0)."""
    raw = X_te.astype(np.float64) @ coef + intercept
    # linear model 出力を 0/1 に thresholding (連続 target を 0.5 で二値化)
    return (raw >= 0.5).astype(int)


def loov_baseline_lr(ds: dict, indices: list[int]) -> dict[str, float]:
    """通常 LR の LOOV phase 平均を測定 (MixedLM との比較用)."""
    out: dict[str, float] = {}
    for phase_name, phases in PHASE_GROUPS.items():
        phase_mask = np.isin(ds["time_phases"], phases)
        if phase_mask.sum() == 0:
            out[phase_name] = 0.0
            continue
        X_p = ds["X"][phase_mask][:, indices]
        y_p = ds["y"][phase_mask]
        v_p = ds["video_ids"][phase_mask]
        accs = _loov_lr_inner(X_p, y_p, v_p)
        out[phase_name] = float(np.mean(accs)) if accs else 0.0
    return out


def _loov_lr_inner(X: np.ndarray, y: np.ndarray, v_ids: np.ndarray) -> list[float]:
    """LR + LOOV 内部ループ."""
    accs: list[float] = []
    uniq = np.unique(v_ids)
    for vid in uniq:
        te_m = v_ids == vid
        tr_m = ~te_m
        if te_m.sum() == 0 or tr_m.sum() == 0:
            continue
        try:
            clf = LogisticRegression(C=LR_C, penalty="l2", max_iter=2000, random_state=RANDOM_SEED)
            clf.fit(X[tr_m], (y[tr_m] > 0).astype(int))
            accs.append(float(clf.score(X[te_m], (y[te_m] > 0).astype(int))))
        except Exception as e:
            print(f"  LOOV LR vid={vid} skip: {e}")
    return accs


def video_holdout_compare(
    ds: dict, indices: list[int]
) -> dict[str, Any]:
    """video holdout で MixedLM (fixed effect) と通常 LR を比較."""
    out: dict[str, Any] = {}
    train_mask, test_mask = video_holdout_split(
        ds["video_ids"], n_test=N_TEST_VIDEOS, seed=RANDOM_SEED
    )
    X_tr_raw = ds["X"][train_mask][:, indices]
    X_te_raw = ds["X"][test_mask][:, indices]
    y_tr = ds["y"][train_mask]
    y_te = ds["y"][test_mask]
    g_tr = ds["video_ids"][train_mask]

    X_tr, mu, sd = standardize(X_tr_raw)
    X_te = (X_te_raw - mu) / sd

    print("[mixed] fitting MixedLM...")
    try:
        _, coef, intercept = fit_mixedlm_train(X_tr, y_tr, g_tr)
        pred_te = predict_mixed(X_te, coef, intercept)
        out["mixedlm_video_holdout"] = float(np.mean(pred_te == (y_te > 0).astype(int)))
        out["mixedlm_n_coef"] = int(coef.size)
    except Exception as e:
        print(f"[mixed] failed: {e}")
        out["mixedlm_video_holdout"] = 0.0
        out["mixedlm_error"] = str(e)

    # 通常 LR baseline (同じ feature subset)
    clf = LogisticRegression(C=LR_C, penalty="l2", max_iter=2000, random_state=RANDOM_SEED)
    clf.fit(X_tr, (y_tr > 0).astype(int))
    out["lr_baseline_video_holdout"] = float(
        clf.score(X_te, (y_te > 0).astype(int))
    )
    return out


def loov_mixedlm_phase(ds: dict, indices: list[int]) -> dict[str, float]:
    """time_phase 別の MixedLM LOOV (動画別 random intercept、test 動画では fixed のみ)."""
    out: dict[str, float] = {}
    for phase_name, phases in PHASE_GROUPS.items():
        phase_mask = np.isin(ds["time_phases"], phases)
        if phase_mask.sum() == 0:
            out[phase_name] = 0.0
            continue
        X_p = ds["X"][phase_mask][:, indices]
        y_p = ds["y"][phase_mask]
        v_p = ds["video_ids"][phase_mask]
        accs = _loov_mixed_inner(X_p, y_p, v_p)
        out[phase_name] = float(np.mean(accs)) if accs else 0.0
    return out


def _loov_mixed_inner(X: np.ndarray, y: np.ndarray, v_ids: np.ndarray) -> list[float]:
    """MixedLM + LOOV 内部ループ."""
    accs: list[float] = []
    uniq = np.unique(v_ids)
    for vid in uniq:
        te_m = v_ids == vid
        tr_m = ~te_m
        if te_m.sum() == 0 or tr_m.sum() == 0:
            continue
        try:
            X_tr_z, mu, sd = standardize(X[tr_m])
            X_te_z = (X[te_m] - mu) / sd
            _, coef, intercept = fit_mixedlm_train(X_tr_z, y[tr_m], v_ids[tr_m])
            pred = predict_mixed(X_te_z, coef, intercept)
            accs.append(float(np.mean(pred == (y[te_m] > 0).astype(int))))
        except Exception as e:
            print(f"  LOOV MIXED vid={vid} skip: {e}")
    return accs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--top-k", type=int, default=TOP_K_FEATURES,
        help=f"MixedLM に渡す特徴量上位数 (default {TOP_K_FEATURES})",
    )
    args = ap.parse_args()

    ds = load_h2_csv(args.csv)
    print(f"[load] n={ds['n']}, d={ds['d']}, videos={len(np.unique(ds['video_ids']))}")

    # MixedLM は計算重いので top-k feature に絞る
    print(f"[select] top {args.top_k} features by |LR coef|")
    indices = select_top_features(ds["X"], ds["y"], args.top_k)
    selected = [ds["feat_cols"][i] for i in indices]
    print(f"[select] selected={selected[:10]}...")

    # 1) video holdout 比較
    print("\n=== video holdout (MixedLM vs LR baseline) ===")
    vh = video_holdout_compare(ds, indices)
    print(f"  MixedLM   vh={vh.get('mixedlm_video_holdout', 0.0):.3f}")
    print(f"  LR base   vh={vh.get('lr_baseline_video_holdout', 0.0):.3f}")

    # 2) LOOV phase 平均比較
    print("\n=== LOOV phase mean (MixedLM vs LR baseline) ===")
    lr_phase = loov_baseline_lr(ds, indices)
    mixed_phase = loov_mixedlm_phase(ds, indices)
    for p in ("start", "mid", "end"):
        print(f"  {p}: LR={lr_phase[p]:.3f} MIXED={mixed_phase[p]:.3f}")
    lr_avg = float(np.mean(list(lr_phase.values())))
    mixed_avg = float(np.mean(list(mixed_phase.values())))
    print(f"  avg: LR={lr_avg:.3f} MIXED={mixed_avg:.3f}")

    payload = {
        "n": ds["n"],
        "d": ds["d"],
        "top_k": args.top_k,
        "selected_features": selected,
        "video_holdout": vh,
        "lr_phase_loov": lr_phase,
        "mixedlm_phase_loov": mixed_phase,
        "lr_phase_avg": lr_avg,
        "mixedlm_phase_avg": mixed_avg,
        "delta_phase_avg": mixed_avg - lr_avg,
        "baseline_h2": {
            "lr_video_holdout": 0.7396,
            "lr_phase_avg": 0.6669,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
