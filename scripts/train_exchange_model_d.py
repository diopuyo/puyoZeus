"""#24 打ち合い計測器「案D」学習器 (2026-08-01)。

## 背景
memory `project_session_2026-07-29_handoff.md` の**案D**
(=期待ダメージ→ΔWinProb のシミュレーションをやめ、実測イベント
exchange_labels.csv から「返される確率・返り量」を直接学習する) を実装する。

## 特徴量 (新規指標追加はしない・33列相当を決め打ちせず動的検出)
exchange_labels.csv の fire_*/opp_*/diff_* 3つ組 (各指標の攻撃側値/相手側値/差分)
+ phase (序/中/終 one-hot) + fire_side (1P/2P one-hot)。base 名は列名から動的に
検出する (手順1: 実カラムを確認してから実装、件数を決め打ちしない)。

## 学習器・評価
- taiou_success (二値・副指標) -> HistGradientBoostingClassifier
- net_ojama_after (連続値・主指標) -> HistGradientBoostingRegressor
  (loss="absolute_error"、外れ値に頑健な MAE 目的関数。net_ojama_after は
  std=242・min=-1138〜max=1862 と裾が重いため squared_error は外れ値に
  引っ張られやすい)
- CV は scripts/exchange_meter_eval_harness.group_kfold_splits (video_id 単位、
  game_idx でのグルーピングは厳禁) を再利用する。
- OOF 予測は scripts/exchange_meter_eval_harness.compare_predictors にそのまま
  渡せる形式で保存する (後の三つ巴比較で再利用するため)。
- train/val の性能ギャップを必ず報告する (小データ想定のため過学習検知が必須)。
- permutation_importance の上位を報告する。

## 使い方
    PYTHONPATH=. python -m scripts.train_exchange_model_d \\
        --labels data/indicators_v2/exchange_labels.csv \\
        --out-dir data/verify/exchange_model_d_2026-08-01

## 注意 (2026-08-01 時点)
最終データ (exchange_labels_regen_step0_2026-08-01.csv) はまだ存在しない。
開発・スモークは既存 exchange_labels.csv (別コーダが Step0 改修中、件数は
実行時点のファイルに依存) で行う。パラメータは最終データが 1000-1500件
規模になる想定で保守的に設定してあるため、現行の大きいデータでは過学習が
出にくく数値は楽観的になりうる (本文注記の通り「暫定」扱いとする)。
"""
from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, roc_auc_score

from scripts.exchange_meter_eval_harness import (
    EXCHANGE_PHASES,
    PredictorPredictions,
    compare_predictors,
    exact_auc,
    group_kfold_splits,
)

# =============================================================================
# 定数
# =============================================================================

# GroupKFold の fold 数 (harness の既定と揃える)
N_FOLDS: int = 5

# HistGBC (taiou_success、副指標) パラメータ
# 小データ (最終想定 1000-1500件) 前提で保守的に (深さ浅め・葉の最小サンプル多め)。
CLS_PARAMS: dict = {
    "max_iter": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "random_state": 42,
    "early_stopping": False,
}

# HistGBR (net_ojama_after、主指標) パラメータ
# loss="absolute_error": net_ojama_after は裾が重い (std=242, max=1862) ため
# 外れ値に頑健な MAE 目的関数を採用する。
REG_PARAMS: dict = {
    "max_iter": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "loss": "absolute_error",
    "random_state": 42,
    "early_stopping": False,
}

# permutation importance
PERM_N_REPEATS: int = 20
PERM_RANDOM_STATE: int = 42
TOP_K_IMPORTANCE: int = 20

# 特徴量対象外の列 (識別子・ターゲット・ターゲット隣接列)
# won は「試合全体の勝敗」で本イベント以降の情報も含みうるため特徴量から除外する
# (task定義の「既存33列相当 + phase one-hot + fire_side」に won は含まれない)。
NON_FEATURE_COLS: frozenset[str] = frozenset([
    "video_id", "game_idx", "t_sec", "fire_side", "phase", "won",
    "net_ojama", "returned", "returned_competitive", "return_window_sec",
    "approx_fire_chains", "opp_buried", "taiou_success", "survived",
    "net_ojama_after",
])


# =============================================================================
# データ読み込み・特徴量構築
# =============================================================================

def load_exchange_labels(csv_path: str) -> pd.DataFrame:
    """exchange_labels.csv を読み込む (行数・動画数をログ出力)。"""
    df = pd.read_csv(csv_path)
    print(f"  読み込み: {len(df)}行  動画数={df['video_id'].nunique()}")
    print(f"  位相内訳: " + ", ".join(
        f"{ph}={int((df['phase'] == ph).sum())}" for ph in EXCHANGE_PHASES
    ))
    return df


def get_indicator_base_names(df: pd.DataFrame) -> list[str]:
    """fire_*/opp_*/diff_* の3つ組が揃っている指標の base 名を動的検出する。

    「33列決め打ち禁止」の手順に従い、実際の列名から機械的に導出する。
    """
    bases: list[str] = []
    for col in df.columns:
        if not col.startswith("fire_"):
            continue
        base = col[len("fire_"):]
        if base in NON_FEATURE_COLS or col in NON_FEATURE_COLS:
            continue
        if f"opp_{base}" in df.columns and f"diff_{base}" in df.columns:
            bases.append(base)
    return sorted(bases)


def build_feature_matrix(
    df: pd.DataFrame, indicator_bases: list[str],
    extra_feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """fire_/opp_/diff_ 3つ組 + phase one-hot + fire_side one-hot の特徴量行列を作る。

    Args:
        extra_feature_cols: 追加でそのまま特徴量に含める列名 (任意、既定 None)。
            三つ巴比較のスタッキング版 (併用) が sim_* 3列を追加するために
            後方互換 (optional 引数のみ) で拡張したもの。既存呼び出し元
            (train_exchange_model_d.main 等) は None のまま呼ぶため挙動は不変。
    """
    cols: list[str] = []
    parts: list[np.ndarray] = []
    for prefix in ("fire_", "opp_", "diff_"):
        for base in indicator_bases:
            col = f"{prefix}{base}"
            cols.append(col)
            parts.append(df[col].astype(float).values)
    for phase in EXCHANGE_PHASES:
        cols.append(f"phase_{phase}")
        parts.append((df["phase"].values == phase).astype(float))
    for side in ("1P", "2P"):
        cols.append(f"fire_side_{side}")
        parts.append((df["fire_side"].values == side).astype(float))
    for col in extra_feature_cols or []:
        cols.append(col)
        parts.append(df[col].astype(float).values)
    X = np.column_stack(parts)
    return X, cols


# =============================================================================
# OOF 学習 (GroupKFold、harness を再利用)
# =============================================================================

def run_oof_classifier(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """taiou_success の GroupKFold OOF 確率予測 + fold毎 train/val AUC ギャップを返す。"""
    oof_proba = np.full(len(y), np.nan)
    fold_rows: list[dict] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(group_kfold_splits(len(y), groups, n_folds)):
        model = HistGradientBoostingClassifier(**CLS_PARAMS)
        model.fit(X[tr_idx], y[tr_idx])
        proba_tr = model.predict_proba(X[tr_idx])[:, 1]
        proba_te = model.predict_proba(X[te_idx])[:, 1]
        oof_proba[te_idx] = proba_te
        auc_tr = exact_auc(y[tr_idx], proba_tr)
        auc_te = exact_auc(y[te_idx], proba_te)
        fold_rows.append({
            "fold": fold_idx + 1, "n_train": len(tr_idx), "n_val": len(te_idx),
            "auc_train": auc_tr, "auc_val": auc_te, "auc_gap(train-val)": auc_tr - auc_te,
        })
        print(f"    [cls] fold{fold_idx + 1}: train={len(tr_idx)} val={len(te_idx)}"
              f" AUC train={auc_tr:.4f} val={auc_te:.4f}")
    return oof_proba, pd.DataFrame(fold_rows)


def run_oof_regressor(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """net_ojama_after の GroupKFold OOF 予測 + fold毎 train/val MAE ギャップを返す。"""
    oof_pred = np.full(len(y), np.nan)
    fold_rows: list[dict] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(group_kfold_splits(len(y), groups, n_folds)):
        model = HistGradientBoostingRegressor(**REG_PARAMS)
        model.fit(X[tr_idx], y[tr_idx])
        pred_tr = model.predict(X[tr_idx])
        pred_te = model.predict(X[te_idx])
        oof_pred[te_idx] = pred_te
        mae_tr = mean_absolute_error(y[tr_idx], pred_tr)
        mae_te = mean_absolute_error(y[te_idx], pred_te)
        fold_rows.append({
            "fold": fold_idx + 1, "n_train": len(tr_idx), "n_val": len(te_idx),
            "mae_train": mae_tr, "mae_val": mae_te, "mae_gap(val-train)": mae_te - mae_tr,
        })
        print(f"    [reg] fold{fold_idx + 1}: train={len(tr_idx)} val={len(te_idx)}"
              f" MAE train={mae_tr:.2f} val={mae_te:.2f}")
    return oof_pred, pd.DataFrame(fold_rows)


# =============================================================================
# Permutation Importance
# =============================================================================

def compute_perm_importance(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int,
    feature_names: list[str], model_ctor, model_params: dict, scoring: str,
) -> pd.DataFrame:
    """fold毎 permutation importance を計算し平均±std を返す (cls/reg 共通処理)。"""
    imp_per_fold: list[np.ndarray] = []
    for tr_idx, te_idx in group_kfold_splits(len(y), groups, n_folds):
        model = model_ctor(**model_params)
        model.fit(X[tr_idx], y[tr_idx])
        perm = permutation_importance(
            model, X[te_idx], y[te_idx],
            n_repeats=PERM_N_REPEATS, random_state=PERM_RANDOM_STATE, scoring=scoring,
        )
        imp_per_fold.append(perm.importances_mean)
    imp_matrix = np.array(imp_per_fold)
    result = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": imp_matrix.mean(axis=0),
        "importance_std": imp_matrix.std(axis=0, ddof=1),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


# =============================================================================
# RT推論用モデル永続化 (--save-model、2026-08-02 追加)
# =============================================================================
# ΔWinProb接続アーキ設計 (案C=仮想盤面2回評価) の Step1: RT (リアルタイム)
# モードでは sim_* 3列のMC計算 (平均1秒/件) が予算オーバーのため、
# 「案D単体 (41特徴量、AUC 0.786/rho 0.694)」をRT層のモデルとして永続化する。
# src/exchange_predictor.py がこの joblib バンドルを読み込む。

def fit_final_models(
    X: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray,
) -> tuple[HistGradientBoostingClassifier, HistGradientBoostingRegressor]:
    """全データで最終モデル (cls+reg) を学習する (OOF評価とは別、RT推論用の本番モデル)。

    OOF学習 (run_oof_classifier/run_oof_regressor) は評価専用でfold毎に別モデルを
    使い捨てるため、RT推論に使う「全データで学習した1本のモデル」はここで別途
    学習する (ハイパラは同一のCLS_PARAMS/REG_PARAMSを再利用、コピペ再実装しない)。
    """
    cls_model = HistGradientBoostingClassifier(**CLS_PARAMS)
    cls_model.fit(X, y_cls)
    reg_model = HistGradientBoostingRegressor(**REG_PARAMS)
    reg_model.fit(X, y_reg)
    return cls_model, reg_model


def save_model_bundle(
    cls_model: HistGradientBoostingClassifier,
    reg_model: HistGradientBoostingRegressor,
    indicator_bases: list[str],
    feature_names: list[str],
    labels_path: str,
    model_date: str,
    n_samples: int,
    save_path: Path,
) -> None:
    """RT推論用モデルバンドルを joblib で保存する (src/exchange_predictor.py が読む形式)。

    src/exchange_predictor.py は scripts/ への依存を持たない設計のため、
    推論時に必要なメタ情報 (indicator_bases・phase一覧・fire_side一覧) を
    全てバンドルに埋め込む (self-contained にする)。
    """
    bundle = {
        "cls_model": cls_model,
        "reg_model": reg_model,
        "indicator_bases": indicator_bases,
        "feature_names": feature_names,
        "phases": EXCHANGE_PHASES,
        "fire_sides": ("1P", "2P"),
        "metadata": {
            "labels_csv": labels_path,
            "model_date": model_date,
            "n_samples": n_samples,
            "cls_params": CLS_PARAMS,
            "reg_params": REG_PARAMS,
        },
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, save_path)
    print(f"  RT推論用モデル保存: {save_path}")


# =============================================================================
# メイン
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する (main を50行以内に保つための分割)。"""
    parser = argparse.ArgumentParser(description="#24 案D 学習器 (taiou_success/net_ojama_after)")
    parser.add_argument("--labels", default="data/indicators_v2/exchange_labels.csv")
    parser.add_argument("--out-dir", default="data/verify/exchange_model_d_2026-08-01")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--save-model", type=Path, default=None,
                         help="RT推論用に全データで学習した最終モデルをjoblib保存するパス"
                              " (既定=None=保存しない、旧挙動と完全一致)")
    parser.add_argument("--model-date", type=str, default=None,
                         help="保存モデルのメタ情報に記録する日時 (既定=実行時刻の自動生成)")
    return parser.parse_args()


def _save_oof_predictions(
    df: pd.DataFrame, oof_proba: np.ndarray, oof_pred: np.ndarray, out_path: Path,
) -> None:
    """後の三つ巴比較で再利用できる形式で OOF 予測を保存する。"""
    out_df = df[["video_id", "game_idx", "t_sec", "phase", "fire_side",
                 "taiou_success", "net_ojama_after"]].copy()
    out_df["prob_taiou_success_oof"] = oof_proba
    out_df["net_ojama_after_oof_pred"] = oof_pred
    out_df.to_csv(out_path, index=False)
    print(f"  OOF予測 保存: {out_path}")


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_exchange_model_d] labels={args.labels}")
    print("\n=== 1. データ読み込み ===")
    df = load_exchange_labels(args.labels)
    indicator_bases = get_indicator_base_names(df)
    print(f"  指標 base 数={len(indicator_bases)} -> 特徴量(fire/opp/diff)="
          f"{len(indicator_bases) * 3}  内訳: {indicator_bases}")

    X, feature_names = build_feature_matrix(df, indicator_bases)
    groups = df["video_id"].values
    y_cls = df["taiou_success"].astype(int).values
    y_reg = df["net_ojama_after"].astype(float).values
    print(f"  特徴量数={len(feature_names)}  サンプル数={len(df)}")

    print("\n=== 2. OOF 学習 (taiou_success, 副指標) ===")
    t0 = time.time()
    oof_proba, fold_metrics_cls = run_oof_classifier(X, y_cls, groups, args.n_folds)
    print(f"  完了 ({time.time() - t0:.1f}秒)")

    print("\n=== 3. OOF 学習 (net_ojama_after, 主指標) ===")
    t0 = time.time()
    oof_pred, fold_metrics_reg = run_oof_regressor(X, y_reg, groups, args.n_folds)
    print(f"  完了 ({time.time() - t0:.1f}秒)")

    fold_metrics_cls.to_csv(out_dir / "train_val_gap_cls.csv", index=False)
    fold_metrics_reg.to_csv(out_dir / "train_val_gap_reg.csv", index=False)
    print(f"  train/valギャップ 平均: AUC gap={fold_metrics_cls['auc_gap(train-val)'].mean():.4f}"
          f"  MAE gap={fold_metrics_reg['mae_gap(val-train)'].mean():.2f}")

    print("\n=== 4. Permutation Importance ===")
    perm_cls = compute_perm_importance(
        X, y_cls, groups, args.n_folds, feature_names,
        HistGradientBoostingClassifier, CLS_PARAMS, "roc_auc",
    )
    perm_reg = compute_perm_importance(
        X, y_reg, groups, args.n_folds, feature_names,
        HistGradientBoostingRegressor, REG_PARAMS, "neg_mean_absolute_error",
    )
    perm_cls.to_csv(out_dir / "permutation_importance_cls.csv", index=False)
    perm_reg.to_csv(out_dir / "permutation_importance_reg.csv", index=False)
    print("  [cls] 上位: " + ", ".join(perm_cls.head(5)["feature"]))
    print("  [reg] 上位: " + ", ".join(perm_reg.head(5)["feature"]))

    print("\n=== 5. OOF 予測保存 + 三つ巴比較ハーネスでレポート出力 ===")
    _save_oof_predictions(df, oof_proba, oof_pred, out_dir / "oof_predictions.csv")
    pred_d = PredictorPredictions(
        name="案D", prob_taiou_success=oof_proba, net_ojama_after_pred=oof_pred,
    )
    compare_predictors(df, [pred_d], out_dir)

    if args.save_model is not None:
        print("\n=== 6. RT推論用モデル永続化 (--save-model) ===")
        model_date = args.model_date or datetime.date.today().isoformat()
        cls_final, reg_final = fit_final_models(X, y_cls, y_reg)
        save_model_bundle(
            cls_final, reg_final, indicator_bases, feature_names,
            args.labels, model_date, len(df), args.save_model,
        )

    print(f"\n出力先: {out_dir}")
    print(f"=== 完了 (入力: {args.labels}) ===")


if __name__ == "__main__":
    main()
