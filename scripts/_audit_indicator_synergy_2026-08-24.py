# -*- coding: utf-8 -*-
"""現行51指標の健全性・位相依存・2指標相乗効果を監査する。

入力は model62_3col の学習CSVと同モデルの特徴量定義・Permutation Importance。
フレーム数の多い試合への偏りを避けるため、相乗効果は
video/game/side/位相ごとの中央値へ集約して評価する。2指標探索は開発動画だけで
行い、探索に使わない確認動画で再評価する。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DEFAULT_CSV = Path(
    "data/verify/labeled_win_model62_3col_2026-08-21/"
    "labeled_win_model62_3col.csv"
)
DEFAULT_MODEL_DIR = Path("data/verify/retrain_model62_3col_2026-08-21")
DEFAULT_OUT_DIR = Path("data/verify/indicator_audit_2026-08-24")
META_COLS = ["video_id", "game_idx", "t_sec", "frame", "tsumo", "side", "won"]
PHASE_NAMES = ("序盤", "中盤", "終盤")
PHASE_BOUNDS = (-0.001, 1.0 / 3.0, 2.0 / 3.0, 1.001)
N_FOLDS = 4
N_JOBS = 3
CONFIRM_MODULO = 5
BOOTSTRAP_REPEATS = 1000
RANDOM_SEED = 20260824
DEAD_AUC_MAX = 0.515
CONDITIONAL_AUC_MIN = 0.530
REDUNDANT_RHO = 0.90


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    if mask.sum() < 20 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], score[mask]))


def _oriented_auc(y: np.ndarray, score: np.ndarray, sign: float | None = None) -> tuple[float, float]:
    raw = _safe_auc(y, score)
    if not np.isfinite(raw):
        return raw, 1.0 if sign is None else sign
    resolved = (1.0 if raw >= 0.5 else -1.0) if sign is None else sign
    return _safe_auc(y, score * resolved), resolved


def _phase_from_progress(progress: pd.Series) -> pd.Categorical:
    return pd.cut(progress, bins=PHASE_BOUNDS, labels=PHASE_NAMES)


def _add_phases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["video_id", "game_idx", "side"]
    max_tsumo = out.groupby(group_cols)["tsumo"].transform("max").replace(0, np.nan)
    out["time_progress_proxy"] = (out["tsumo"] / max_tsumo).fillna(0.0).clip(0.0, 1.0)
    out["phase_time_proxy"] = _phase_from_progress(out["time_progress_proxy"])
    out["phase_board_state"] = _phase_from_progress(out["match_progress"])
    return out


def _load_inputs(csv_path: Path, model_dir: Path) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    features = json.loads((model_dir / "feature_cols_full.json").read_text(encoding="utf-8"))
    usecols = list(dict.fromkeys(META_COLS + ["match_progress"] + features))
    df = pd.read_csv(csv_path, usecols=usecols)
    df = df[df["won"].isin([0.0, 1.0])].copy()
    df["won"] = df["won"].astype(np.int8)
    df = _add_phases(df)
    perm = pd.read_csv(model_dir / "permutation_importance_full.csv")
    perm = perm.rename(columns={"feature": "indicator"})
    return df, features, perm


def _video_auc_values(df: pd.DataFrame, col: str, sign: float) -> list[float]:
    values: list[float] = []
    for _, part in df.groupby("video_id", observed=True):
        auc = _safe_auc(part["won"].to_numpy(), part[col].to_numpy(float) * sign)
        if np.isfinite(auc):
            values.append(auc)
    return values


def _phase_aucs(df: pd.DataFrame, col: str, phase_col: str, sign: float) -> dict[str, float]:
    result: dict[str, float] = {}
    for phase in PHASE_NAMES:
        part = df[df[phase_col] == phase]
        result[phase] = _safe_auc(
            part["won"].to_numpy(), part[col].to_numpy(float) * sign,
        )
    return result


def _feature_health_rows(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y = df["won"].to_numpy()
    for col in features:
        series = df[col].astype(float)
        overall_auc, sign = _oriented_auc(y, series.to_numpy())
        video_aucs = _video_auc_values(df, col, sign)
        time_auc = _phase_aucs(df, col, "phase_time_proxy", sign)
        board_auc = _phase_aucs(df, col, "phase_board_state", sign)
        finite = series[np.isfinite(series)]
        rows.append(_health_row(col, finite, overall_auc, sign, video_aucs, time_auc, board_auc, series))
    return pd.DataFrame(rows)


def _health_row(
    col: str, finite: pd.Series, overall_auc: float, sign: float,
    video_aucs: list[float], time_auc: dict[str, float], board_auc: dict[str, float],
    full_series: pd.Series,
) -> dict[str, object]:
    row: dict[str, object] = {
        "indicator": col, "direction": "+" if sign > 0 else "-",
        "overall_auc": overall_auc,
        "video_median_auc": float(np.median(video_aucs)) if video_aucs else np.nan,
        "video_auc_iqr": float(np.subtract(*np.percentile(video_aucs, [75, 25]))) if video_aucs else np.nan,
        "missing_rate": float(full_series.isna().mean()), "n_unique": int(finite.nunique()),
        "std": float(finite.std(ddof=0)), "zero_rate": float((finite == 0).mean()),
        "min": float(finite.min()), "median": float(finite.median()), "max": float(finite.max()),
    }
    for phase in PHASE_NAMES:
        row[f"time_{phase}_auc"] = time_auc[phase]
        row[f"board_{phase}_auc"] = board_auc[phase]
    phase_values = [v for v in time_auc.values() if np.isfinite(v)]
    row["time_phase_auc_range"] = max(phase_values) - min(phase_values) if phase_values else np.nan
    row["max_phase_auc"] = max(phase_values) if phase_values else np.nan
    return row


def _add_redundancy(health: pd.DataFrame, df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = df[features].corr(method="spearman")
    rows: list[dict[str, object]] = []
    max_corr: dict[str, tuple[str, float]] = {c: ("", 0.0) for c in features}
    for i, left in enumerate(features):
        for right in features[i + 1:]:
            rho = float(corr.at[left, right])
            if not np.isfinite(rho):
                continue
            if abs(rho) > abs(max_corr[left][1]):
                max_corr[left] = (right, rho)
            if abs(rho) > abs(max_corr[right][1]):
                max_corr[right] = (left, rho)
            if abs(rho) >= REDUNDANT_RHO:
                rows.append({"indicator_a": left, "indicator_b": right, "spearman_rho": rho})
    health["most_correlated_with"] = health["indicator"].map(lambda c: max_corr[c][0])
    health["max_abs_spearman"] = health["indicator"].map(lambda c: abs(max_corr[c][1]))
    return health, pd.DataFrame(rows).sort_values("spearman_rho", key=abs, ascending=False)


def _classify_health(health: pd.DataFrame, perm: pd.DataFrame) -> pd.DataFrame:
    out = health.merge(perm, on="indicator", how="left")
    structural = (out["n_unique"] <= 1) | (out["std"] <= 1e-12)
    coverage = (out["missing_rate"] >= 0.99) & ~structural
    weak_all = (out["overall_auc"] <= DEAD_AUC_MAX) & (out["max_phase_auc"] <= DEAD_AUC_MAX)
    model_dead = out["importance_mean"].fillna(0.0) <= 0.0
    conditional = (out["overall_auc"] <= DEAD_AUC_MAX) & (out["max_phase_auc"] >= CONDITIONAL_AUC_MIN)
    redundant = ((out["max_abs_spearman"] >= REDUNDANT_RHO)
                 & (out["importance_mean"].fillna(0.0) <= 0.0002))
    out["is_dead_candidate"] = model_dead & weak_all
    out["is_phase_conditional"] = conditional
    out["is_redundant_low_increment"] = redundant
    out["is_coverage_failure"] = coverage
    out["health_class"] = "weak"
    out.loc[~weak_all & ~model_dead, "health_class"] = "active"
    out.loc[redundant, "health_class"] = "redundant_low_increment"
    out.loc[model_dead & weak_all, "health_class"] = "dead_candidate"
    out.loc[conditional, "health_class"] = "phase_conditional"
    out.loc[coverage, "health_class"] = "coverage_failure"
    out.loc[structural, "health_class"] = "structurally_dead"
    return out.sort_values(["health_class", "overall_auc"], ascending=[True, False])


def _aggregate_game_phase(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    keys = ["video_id", "game_idx", "side", "phase_time_proxy"]
    numeric = features + ["won", "time_progress_proxy", "match_progress"]
    agg = df.groupby(keys, observed=True)[numeric].median().reset_index()
    agg = agg.rename(columns={"phase_time_proxy": "phase"})
    agg["won"] = agg["won"].round().astype(np.int8)
    return agg


def _split_videos(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    videos = sorted(df["video_id"].unique())
    confirm = set(v for i, v in enumerate(videos) if i % CONFIRM_MODULO == CONFIRM_MODULO - 1)
    confirm_mask = df["video_id"].isin(confirm).to_numpy()
    return ~confirm_mask, confirm_mask


def _design(train: np.ndarray, test: np.ndarray, interaction: bool) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train)
    test_z = scaler.transform(test)
    if interaction and train_z.shape[1] == 2:
        train_z = np.column_stack([train_z, train_z[:, 0] * train_z[:, 1]])
        test_z = np.column_stack([test_z, test_z[:, 0] * test_z[:, 1]])
    return train_z, test_z


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, interaction: bool) -> np.ndarray:
    train_z, test_z = _design(train_x, test_x, interaction)
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=200, random_state=RANDOM_SEED)
    model.fit(train_z, train_y)
    return model.predict_proba(test_z)[:, 1]


def _oof_predictions(df: pd.DataFrame, cols: list[str], interaction: bool) -> np.ndarray:
    x = df[cols].fillna(0.0).to_numpy(float)
    y = df["won"].to_numpy()
    groups = df["video_id"].to_numpy()
    pred = np.full(len(df), np.nan)
    splitter = GroupKFold(n_splits=min(N_FOLDS, np.unique(groups).size))
    for train_idx, test_idx in splitter.split(x, y, groups):
        pred[test_idx] = _fit_predict(x[train_idx], y[train_idx], x[test_idx], interaction)
    return pred


def _single_oof(df: pd.DataFrame, features: list[str]) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    aucs: dict[str, float] = {}
    predictions: dict[str, np.ndarray] = {}
    y = df["won"].to_numpy()
    for col in features:
        pred = _oof_predictions(df, [col], interaction=False)
        predictions[col] = pred
        aucs[col] = _safe_auc(y, pred)
    return aucs, predictions


def _evaluate_pair(
    dev: pd.DataFrame, left: str, right: str,
    single_auc: dict[str, float], y: np.ndarray,
) -> dict[str, object]:
    add_pred = _oof_predictions(dev, [left, right], interaction=False)
    int_pred = _oof_predictions(dev, [left, right], interaction=True)
    additive_auc = _safe_auc(y, add_pred)
    interaction_auc = _safe_auc(y, int_pred)
    best_col = left if single_auc[left] >= single_auc[right] else right
    best_single = single_auc[best_col]
    return {
        "indicator_a": left, "indicator_b": right,
        "dev_best_single_indicator": best_col,
        "dev_best_single_auc": best_single, "dev_additive_auc": additive_auc,
        "dev_interaction_auc": interaction_auc,
        "dev_gain_vs_best_single": interaction_auc - best_single,
        "dev_interaction_gain": interaction_auc - additive_auc,
    }


def _screen_pairs(dev: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    single_auc, single_pred = _single_oof(dev, features)
    y = dev["won"].to_numpy()
    pairs = [(left, right) for i, left in enumerate(features) for right in features[i + 1:]]
    rows = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(_evaluate_pair)(dev, left, right, single_auc, y) for left, right in pairs
    )
    ranked = pd.DataFrame(rows).sort_values("dev_gain_vs_best_single", ascending=False)
    return ranked, single_pred


def _phase_metrics(y: np.ndarray, pred: np.ndarray, phase: np.ndarray) -> dict[str, float]:
    return {name: _safe_auc(y[phase == name], pred[phase == name]) for name in PHASE_NAMES}


def _bootstrap_video_gain(
    df: pd.DataFrame, pair_pred: np.ndarray, single_pred: np.ndarray,
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    videos = np.unique(df["video_id"].to_numpy())
    indices = {v: np.flatnonzero(df["video_id"].to_numpy() == v) for v in videos}
    gains: list[float] = []
    y = df["won"].to_numpy()
    for _ in range(BOOTSTRAP_REPEATS):
        sampled = rng.choice(videos, size=len(videos), replace=True)
        idx = np.concatenate([indices[v] for v in sampled])
        gain = _safe_auc(y[idx], pair_pred[idx]) - _safe_auc(y[idx], single_pred[idx])
        if np.isfinite(gain):
            gains.append(gain)
    return tuple(float(v) for v in np.percentile(gains, [2.5, 97.5]))


def _confirm_top_pairs(
    dev: pd.DataFrame, confirm: pd.DataFrame, ranked: pd.DataFrame, top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y_dev, y_con = dev["won"].to_numpy(), confirm["won"].to_numpy()
    phase = confirm["phase"].astype(str).to_numpy()
    for _, pair in ranked.head(top_n).iterrows():
        left, right = pair["indicator_a"], pair["indicator_b"]
        x_dev = dev[[left, right]].fillna(0.0).to_numpy(float)
        x_con = confirm[[left, right]].fillna(0.0).to_numpy(float)
        additive_pred = _fit_predict(x_dev, y_dev, x_con, interaction=False)
        pair_pred = _fit_predict(x_dev, y_dev, x_con, interaction=True)
        single_preds = {
            col: _fit_predict(dev[[col]].fillna(0.0).to_numpy(float), y_dev,
                              confirm[[col]].fillna(0.0).to_numpy(float), False)
            for col in (left, right)
        }
        best_col = str(pair["dev_best_single_indicator"])
        rows.append(_confirm_row(
            pair, confirm, pair_pred, additive_pred, single_preds[best_col], best_col, phase,
        ))
    return pd.DataFrame(rows).sort_values("confirm_gain_vs_best_single", ascending=False)


def _confirm_row(
    pair: pd.Series, confirm: pd.DataFrame, pair_pred: np.ndarray, additive_pred: np.ndarray,
    single_pred: np.ndarray, best_col: str, phase: np.ndarray,
) -> dict[str, object]:
    y = confirm["won"].to_numpy()
    pair_phase = _phase_metrics(y, pair_pred, phase)
    single_phase = _phase_metrics(y, single_pred, phase)
    lo, hi = _bootstrap_video_gain(confirm, pair_pred, single_pred)
    int_lo, int_hi = _bootstrap_video_gain(confirm, pair_pred, additive_pred)
    row = pair.to_dict()
    row.update({
        "confirm_best_single": best_col,
        "confirm_best_single_auc": _safe_auc(y, single_pred),
        "confirm_additive_auc": _safe_auc(y, additive_pred),
        "confirm_interaction_auc": _safe_auc(y, pair_pred),
        "confirm_gain_vs_best_single": _safe_auc(y, pair_pred) - _safe_auc(y, single_pred),
        "confirm_gain_ci95_low": lo, "confirm_gain_ci95_high": hi,
        "confirm_interaction_gain": _safe_auc(y, pair_pred) - _safe_auc(y, additive_pred),
        "confirm_interaction_ci95_low": int_lo, "confirm_interaction_ci95_high": int_hi,
    })
    for name in PHASE_NAMES:
        row[f"confirm_{name}_auc"] = pair_phase[name]
        row[f"confirm_{name}_gain"] = pair_phase[name] - single_phase[name]
    return row


def _summary(
    df: pd.DataFrame, agg: pd.DataFrame, health: pd.DataFrame,
    pairs: pd.DataFrame, interaction_pairs: pd.DataFrame, confirm_videos: Iterable[str],
) -> dict[str, object]:
    classes = health["health_class"].value_counts().to_dict()
    positive = pairs[pairs["confirm_gain_ci95_low"] > 0.0]
    positive_interaction = interaction_pairs[
        interaction_pairs["confirm_interaction_ci95_low"] > 0.0
    ]
    return {
        "input_rows_labeled": int(len(df)), "videos": int(df["video_id"].nunique()),
        "games": int(df[["video_id", "game_idx"]].drop_duplicates().shape[0]),
        "game_phase_rows": int(len(agg)), "features": int(len(health)),
        "health_class_counts": {str(k): int(v) for k, v in classes.items()},
        "confirm_videos": sorted(str(v) for v in confirm_videos),
        "confirmed_positive_pair_count_top20": int(len(positive)),
        "confirmed_positive_nonlinear_count_top20": int(len(positive_interaction)),
        "best_confirmed_pair": positive.iloc[0].to_dict() if len(positive) else None,
        "best_confirmed_nonlinear_pair": (
            positive_interaction.sort_values("confirm_interaction_gain", ascending=False).iloc[0].to_dict()
            if len(positive_interaction) else None
        ),
        "phase_caveat": "time phase uses approximate tsumo; board-state phase is non-monotonic",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="現行指標の死活・位相・相乗効果監査")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-pairs", type=int, default=20)
    parser.add_argument("--reuse-screen", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df, features, perm = _load_inputs(args.csv, args.model_dir)
    health = _feature_health_rows(df, features)
    health, redundant = _add_redundancy(health, df, features)
    health = _classify_health(health, perm)
    agg = _aggregate_game_phase(df, features)
    dev_mask, confirm_mask = _split_videos(agg)
    dev, confirm = agg[dev_mask].reset_index(drop=True), agg[confirm_mask].reset_index(drop=True)
    screen_path = args.out_dir / "pair_screen_development.csv"
    if args.reuse_screen and screen_path.exists():
        ranked = pd.read_csv(screen_path)
    else:
        ranked, _ = _screen_pairs(dev, features)
    confirmed = _confirm_top_pairs(dev, confirm, ranked, args.top_pairs)
    nonlinear_ranked = ranked.sort_values("dev_interaction_gain", ascending=False)
    nonlinear_confirmed = _confirm_top_pairs(dev, confirm, nonlinear_ranked, args.top_pairs)
    health.to_csv(args.out_dir / "feature_health.csv", index=False)
    redundant.to_csv(args.out_dir / "redundant_pairs.csv", index=False)
    ranked.to_csv(args.out_dir / "pair_screen_development.csv", index=False)
    confirmed.to_csv(args.out_dir / "pair_confirmation_top20.csv", index=False)
    nonlinear_confirmed.to_csv(
        args.out_dir / "pair_confirmation_nonlinear_top20.csv", index=False,
    )
    summary = _summary(
        df, agg, health, confirmed, nonlinear_confirmed, confirm["video_id"].unique(),
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
