"""#24 頑健性チェック: 終局イベント合成込みラベルでの三つ巴再学習 (2026-08-03 main発注)。

合成込み aug ラベル (18,057件、is_synthetic_terminal_event列で明示) で
併用スタッキングを再学習し、旧確定値 (16,470件、rho=0.8077/AUC=0.8366、
data/verify/exchange_triple_comparison_step3_2026-08-02) と比較する。

構成2種類:
    (i)  合成行込み学習 (18,057件全部でGroupKFold OOF)
    (ii) 合成行除外学習 (is_synthetic_terminal_event=0の16,470件のみで
         GroupKFold OOF) + 合成行は「学習に一度も使われていない最終モデル」
         で予測 (完全な held-out 評価、リーク無し)

既存資産の再利用のみ (再実装禁止): train_exchange_model_d.py の
build_feature_matrix/run_oof_classifier/run_oof_regressor/fit_final_models、
exchange_meter_eval_harness.py の build_scope_comparison_table、
run_exchange_triple_comparison.py の build_stacking_oof_predictions(構成i用)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.exchange_meter_eval_harness import PredictorPredictions, build_scope_comparison_table
from scripts.train_exchange_model_d import (
    N_FOLDS,
    build_feature_matrix,
    fit_final_models,
    get_indicator_base_names,
    run_oof_classifier,
    run_oof_regressor,
)
from scripts.run_exchange_triple_comparison import SIM_FEATURE_COLS, build_stacking_oof_predictions

AUG_CSV = Path("data/indicators_v2/exchange_labels_regen_synth_aug_2026-08-03.csv")
OUT_DIR = Path("data/verify/exchange_synth_robustness_2026-08-03")


def train_on_subset_eval_on_all(
    df_all: pd.DataFrame, train_mask: np.ndarray,
    extra_feature_cols: "list[str] | None" = None, n_folds: int = N_FOLDS,
) -> tuple[np.ndarray, np.ndarray]:
    """train_mask行のみでGroupKFold OOFし、train_mask=Falseの行はtrain_mask
    行全体で学習した最終モデル (完全に held-out、リーク無し) で予測する。

    戻り値は df_all と同じ行順・長さの (prob_taiou_success, net_ojama_after_pred)。
    """
    indicator_bases = get_indicator_base_names(df_all)
    X_all, _ = build_feature_matrix(df_all, indicator_bases, extra_feature_cols=extra_feature_cols)
    y_cls_all = df_all["taiou_success"].astype(int).values
    y_reg_all = df_all["net_ojama_after"].astype(float).values
    groups_all = df_all["video_id"].values

    X_tr, y_cls_tr, y_reg_tr, groups_tr = (
        X_all[train_mask], y_cls_all[train_mask], y_reg_all[train_mask], groups_all[train_mask])
    oof_proba_tr, _ = run_oof_classifier(X_tr, y_cls_tr, groups_tr, n_folds)
    oof_pred_tr, _ = run_oof_regressor(X_tr, y_reg_tr, groups_tr, n_folds)
    cls_final, reg_final = fit_final_models(X_tr, y_cls_tr, y_reg_tr)

    holdout_mask = ~train_mask
    proba_all = np.full(len(df_all), np.nan)
    pred_all = np.full(len(df_all), np.nan)
    proba_all[train_mask] = oof_proba_tr
    pred_all[train_mask] = oof_pred_tr
    if holdout_mask.any():
        proba_all[holdout_mask] = cls_final.predict_proba(X_all[holdout_mask])[:, 1]
        pred_all[holdout_mask] = reg_final.predict(X_all[holdout_mask])
    return proba_all, pred_all


def build_predictors_for_construction(
    df: pd.DataFrame, d_proba: np.ndarray, d_pred: np.ndarray,
    stack_proba: np.ndarray, stack_pred: np.ndarray,
) -> list[PredictorPredictions]:
    """3予測器 (案D/修正シミュ/併用) の PredictorPredictions を組み立てる。"""
    return [
        PredictorPredictions(name="案D", prob_taiou_success=d_proba, net_ojama_after_pred=d_pred),
        PredictorPredictions(name="修正シミュ", prob_taiou_success=1.0 - df["sim_damage_score"].values,
                              net_ojama_after_pred=df["sim_damage_score"].values),
        PredictorPredictions(name="併用(スタッキング)", prob_taiou_success=stack_proba,
                              net_ojama_after_pred=stack_pred),
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(AUG_CSV)
    n_synth = int(df["is_synthetic_terminal_event"].sum())
    print(f"=== 入力: {len(df)}行 (動画数={df['video_id'].nunique()}, 合成行={n_synth}"
          f" [{n_synth / len(df):.1%}]) ===")

    print("\n=== 構成(i): 合成行込み学習 ===")
    indicator_bases = get_indicator_base_names(df)
    X_d, _ = build_feature_matrix(df, indicator_bases)
    groups = df["video_id"].values
    y_cls, y_reg = df["taiou_success"].astype(int).values, df["net_ojama_after"].astype(float).values
    d_proba_i, _ = run_oof_classifier(X_d, y_cls, groups, N_FOLDS)
    d_pred_i, _ = run_oof_regressor(X_d, y_reg, groups, N_FOLDS)
    stack_proba_i, stack_pred_i, _ = build_stacking_oof_predictions(df, N_FOLDS)
    predictors_i = build_predictors_for_construction(df, d_proba_i, d_pred_i, stack_proba_i, stack_pred_i)
    table_i = build_scope_comparison_table(df, predictors_i)
    table_i.insert(0, "構成", "(i)合成込み学習")
    table_i.to_csv(OUT_DIR / "table_construction_i.csv", index=False)
    print(table_i.to_string(index=False))

    print("\n=== 構成(ii): 合成行除外学習 + 評価は全行 ===")
    train_mask = (df["is_synthetic_terminal_event"] == 0).values
    d_proba_ii, d_pred_ii = train_on_subset_eval_on_all(df, train_mask, extra_feature_cols=None)
    stack_proba_ii, stack_pred_ii = train_on_subset_eval_on_all(
        df, train_mask, extra_feature_cols=list(SIM_FEATURE_COLS))
    predictors_ii = build_predictors_for_construction(
        df, d_proba_ii, d_pred_ii, stack_proba_ii, stack_pred_ii)
    table_ii = build_scope_comparison_table(df, predictors_ii)
    table_ii.insert(0, "構成", "(ii)合成除外学習")
    table_ii.to_csv(OUT_DIR / "table_construction_ii.csv", index=False)
    print(table_ii.to_string(index=False))

    print("\n=== 参考値: 合成行のみの層別 (構成iiのheld-out予測を使用) ===")
    synth_mask = ~train_mask
    if synth_mask.any():
        df_synth = df.loc[synth_mask].reset_index(drop=True)
        preds_synth = build_predictors_for_construction(
            df_synth, d_proba_ii[synth_mask], d_pred_ii[synth_mask],
            stack_proba_ii[synth_mask], stack_pred_ii[synth_mask])
        table_synth = build_scope_comparison_table(df_synth, preds_synth)
        table_synth.to_csv(OUT_DIR / "table_synthetic_only.csv", index=False)
        print(table_synth.to_string(index=False))

    combined = pd.concat([table_i, table_ii], ignore_index=True)
    combined.to_csv(OUT_DIR / "table_combined.csv", index=False)
    print(f"\n[保存] {OUT_DIR}")


if __name__ == "__main__":
    main()
