"""#24 打ち合い計測器「三つ巴比較」駆動スクリプト (2026-08-02)。

案D (実データ学習モデル) / 修正シミュ (sim_damage_score) / スタッキング版 (併用)
の3予測器を同一 held-out イベント集合上で比較する。本スクリプト自体は指標定義を
持たず、既存資産の組み合わせのみで構成する (再実装禁止):

    - scripts/train_exchange_model_d.py: 特徴量構築・OOF学習(GroupKFold)・
      ハイパラ (再利用、コピペ再実装しない)。
    - scripts/exchange_meter_eval_harness.py: 比較 API (compare_predictors)。
    - scripts/augment_exchange_labels_with_sim.py: sim_* 3列付き aug CSV
      (本スクリプトの入力、既に生成済みのファイルを読むだけ)。

## 処理手順
    1. aug CSV (sim_* 3列付き) を読込む。
    2. 案D の OOF 予測 (data/verify/<model-d-dir>/oof_predictions.csv) を
       複合キー (video_id, game_idx, t_sec, fire_side) で aug CSV に突合する。
       突合失敗行 (どちらか一方にのみ存在) は比較対象から除外し、件数を必ずログする。
    3. sim_damage_score が NaN の行を除外する (3予測器を同一行集合で比較する
       ため)。除外数を必ずログする。
    4. 「併用」= 案Dの特徴量 (fire_/opp_/diff_ 3つ組 + phase/fire_side one-hot)
       + sim_* 3列 を train_exchange_model_d と同一の fold 分割・ハイパラで
       学習し OOF 予測を得る (train_exchange_model_d.build_feature_matrix の
       optional 引数 extra_feature_cols で拡張、コピペ再実装しない)。
    5. compare_predictors() で3予測器のレポート (comparison_report.md +
       reliability_diagrams.png) を出力する。

## 修正シミュの符号規約について (2026-08-02 main精査で実バグ確定・修正済み)
sim_damage_score (scripts/measure_exchange_effectiveness.estimate_expected_net_damage
の返り値) は旧実装で ojama_damage(attacker_board_after_fire, ...) と
**攻撃側自身の盤面**で威力評価しており実バグだった (user伝授ドメインルール
reference_ojama_damage_nonlinear_2026-07-29「威力は受け側の残り容量に依存」に反する)。
ojama_damage(opp_board, ...) (受け側=相手の盤面) に修正済み。修正後の定義は
「0〜1、大きいほど攻撃側に有利 (相手が受ける期待正味ダメージ)」であり、
net_ojama_after (「攻撃側が届けた正味ダメージ、大きいほど攻撃側に有利」) と
**同じ向き**になった。一方 taiou_success (=1: 受け手/対応側が成功) は
net_ojama_after/sim_damage_score が高いほど不利な側の指標のため、
prob_taiou_success には sim_damage_score をそのまま使わず 1 - sim_damage_score
(符号反転) を使う (build_predictor_set 参照)。

_log_sign_diagnostics は上記の想定符号 (sim_damage_score vs net_ojama_after は
**正相関**、sim_damage_score vs taiou_success は**負相関**) からの逸脱を
警告する (「正相関=健全」は net_ojama_after との比較の話であり、
taiou_success とは逆符号が正しい設計であることに注意)。

## 使い方 (本走行、入力ファイルが揃ってから実行すること)
    PYTHONPATH=. python -m scripts.run_exchange_triple_comparison \\
        --aug-csv data/indicators_v2/exchange_labels_regen_step0_aug_2026-08-01.csv \\
        --model-d-dir data/verify/exchange_model_d_regen_2026-08-02 \\
        --out-dir data/verify/exchange_triple_comparison_2026-08-02

## 動作確認 (--limit、入力がまだ生成中の間の軽量チェック用)
    PYTHONPATH=. python -m scripts.run_exchange_triple_comparison --limit 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.exchange_meter_eval_harness import PredictorPredictions, compare_predictors
from scripts.train_exchange_model_d import (
    N_FOLDS,
    build_feature_matrix,
    get_indicator_base_names,
    load_exchange_labels,
    run_oof_classifier,
    run_oof_regressor,
)

# =============================================================================
# 定数
# =============================================================================

DEFAULT_AUG_CSV = Path("data/indicators_v2/exchange_labels_regen_step0_aug_2026-08-01.csv")
DEFAULT_MODEL_D_DIR = Path("data/verify/exchange_model_d_regen_2026-08-02")
DEFAULT_OUT_DIR = Path("data/verify/exchange_triple_comparison_2026-08-02")

# 案D OOF 出力 (oof_predictions.csv) と aug CSV を突合する複合キー
# (game_idx でのグルーピング/突合は video_id 内 1P/2P 独立カウンタでズレる
# 前科があるため、単独の game_idx でなく video_id+t_sec+fire_side も併用する)。
MERGE_KEYS: tuple[str, ...] = ("video_id", "game_idx", "t_sec", "fire_side")

# 案D OOF 出力から突合に使う列 (これ以外は aug 側の値をそのまま使う)
MODEL_D_OOF_PRED_COLS: tuple[str, ...] = ("prob_taiou_success_oof", "net_ojama_after_oof_pred")

# 修正シミュ・スタッキング特徴量として使う sim_* 3列 (aug CSV で付与済み)
SIM_FEATURE_COLS: tuple[str, ...] = ("sim_k_hands", "sim_expected_counter_ojama", "sim_damage_score")

# 「修正シミュ」予測器・NaN除外の基準とする列
SIM_SCORE_COL: str = "sim_damage_score"


# =============================================================================
# 1. 入力読込
# =============================================================================

def load_model_d_oof(model_d_dir: Path) -> pd.DataFrame:
    """案D の OOF 予測 CSV (oof_predictions.csv) を読み込む。"""
    oof_path = model_d_dir / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"案D OOF予測が見つかりません: {oof_path}")
    df = pd.read_csv(oof_path)
    print(f"  案D OOF読込: {len(df)}行 ({oof_path})")
    return df


# =============================================================================
# 2. 突合 (aug CSV <-> 案D OOF)
# =============================================================================

def _raise_if_duplicate_keys(df: pd.DataFrame, merge_keys: tuple[str, ...], label: str) -> None:
    """複合キーの重複を検出したら例外を送出する (暗黙のクロス積結合を避ける安全ガード)。"""
    dup_count = int(df.duplicated(subset=list(merge_keys)).sum())
    if dup_count > 0:
        raise ValueError(
            f"{label} に複合キー {merge_keys} の重複が {dup_count} 件あります。"
            "突合前に重複を解消してください。",
        )


def align_aug_with_model_d(
    aug_df: pd.DataFrame, oof_df: pd.DataFrame, merge_keys: tuple[str, ...] = MERGE_KEYS,
) -> pd.DataFrame:
    """aug CSV (sim_*列付き) と案D OOF予測を複合キーで突合する。

    突合に失敗した行 (どちらか一方にのみ存在するキー) は比較対象から除外し、
    件数を必ずログする (silent drop 禁止)。重複キーは危険な暗黙のクロス積を
    避けるため例外を送出する。
    """
    _raise_if_duplicate_keys(aug_df, merge_keys, "aug CSV")
    _raise_if_duplicate_keys(oof_df, merge_keys, "案D OOF")

    bring_cols = list(merge_keys) + list(MODEL_D_OOF_PRED_COLS)
    merged = aug_df.merge(oof_df[bring_cols], on=list(merge_keys), how="inner", validate="one_to_one")
    n_aug_only = len(aug_df) - len(merged)
    n_oof_only = len(oof_df) - len(merged)
    print(f"  突合結果: 一致={len(merged)}行  aug側のみ(突合失敗)={n_aug_only}行"
          f"  案D側のみ(突合失敗)={n_oof_only}行")
    return merged


# =============================================================================
# 3. sim_damage_score NaN 除外
# =============================================================================

def filter_nan_sim_rows(df: pd.DataFrame, sim_col: str = SIM_SCORE_COL) -> pd.DataFrame:
    """sim_col が NaN の行を除外する (3予測器を同一行集合で比較するため)。

    除外数を必ずログする (silent drop 禁止)。
    """
    n_before = len(df)
    filtered = df.loc[df[sim_col].notna()].reset_index(drop=True)
    n_excluded = n_before - len(filtered)
    print(f"  {sim_col} NaN除外: {n_excluded}/{n_before}行 除外 (残り{len(filtered)}行)")
    return filtered


def _log_sign_diagnostics(df: pd.DataFrame, sim_col: str = SIM_SCORE_COL) -> None:
    """sim_damage_score と正解ラベルの相関符号を診断ログ出力する (測定器事故対策)。

    2026-08-02 バグ修正 (estimate_expected_net_damage の評価基準を opp_board に
    修正) 後の想定符号:
        - sim_damage_score vs net_ojama_after: **正相関** が健全
          (両者とも「攻撃側が相手に与えた正味ダメージ、大きいほど攻撃側に有利」
          で同じ向き)。
        - sim_damage_score vs taiou_success: **負相関** が健全
          (taiou_success=1 は受け手/対応側の成功、sim_damage_score が高い=
          攻撃側に有利=受け手の失敗寄りのため、向きが逆になるのが正しい)。
    上記想定から外れた符号が出た場合は警告し、比較結果を鵜呑みにせず
    user/architect レビューを仰ぐ (feedback_consult_user_puyo_domain.md 準拠)。
    """
    rho_net = float(spearmanr(df[sim_col], df["net_ojama_after"]).statistic)
    rho_taiou = float(spearmanr(df[sim_col], df["taiou_success"]).statistic)
    print(f"  [符号診断] {sim_col} vs net_ojama_after  spearman_rho={rho_net:.3f} (想定: 正相関=健全)")
    print(f"  [符号診断] {sim_col} vs taiou_success     spearman_rho={rho_taiou:.3f} (想定: 負相関=健全)")
    if rho_net < 0.0 or rho_taiou > 0.0:
        print("  ⚠️ 符号確認が必要: 想定と逆符号の可能性。三つ巴比較レポートの解釈前に"
              "user/architectへ確認すること (feedback_consult_user_puyo_domain.md 準拠)。")


# =============================================================================
# 4. スタッキング版 (併用) OOF 学習
# =============================================================================

def build_stacking_oof_predictions(
    df: pd.DataFrame, n_folds: int = N_FOLDS,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """案Dの特徴量 + sim_* 3列 で train_exchange_model_d と同一 fold・ハイパラで
    スタッキング版 (併用) の OOF 予測を作る (関数を import して再利用、コピペ禁止)。
    """
    indicator_bases = get_indicator_base_names(df)
    X, feature_names = build_feature_matrix(df, indicator_bases, extra_feature_cols=list(SIM_FEATURE_COLS))
    groups = df["video_id"].values
    y_cls = df["taiou_success"].astype(int).values
    y_reg = df["net_ojama_after"].astype(float).values
    print(f"  スタッキング特徴量数={len(feature_names)}"
          f" (案D由来{len(feature_names) - len(SIM_FEATURE_COLS)}+sim{len(SIM_FEATURE_COLS)})")
    oof_proba, _fold_cls = run_oof_classifier(X, y_cls, groups, n_folds)
    oof_pred, _fold_reg = run_oof_regressor(X, y_reg, groups, n_folds)
    return oof_proba, oof_pred, feature_names


# =============================================================================
# 5. 予測器セット組み立て + 比較実行
# =============================================================================

def build_predictor_set(
    df: pd.DataFrame, oof_proba_stack: np.ndarray, oof_pred_stack: np.ndarray,
) -> list[PredictorPredictions]:
    """3予測器 (案D/修正シミュ/併用) の PredictorPredictions を組み立てる。

    修正シミュの prob_taiou_success には sim_damage_score をそのまま使わず
    1 - sim_damage_score (符号反転) を使う。2026-08-02のバグ修正で
    sim_damage_score は「大きいほど攻撃側に有利 (相手が受ける正味ダメージ)」
    という net_ojama_after と同じ向きになったが、taiou_success (=1: 受け手の
    対応成功) は攻撃側有利とは逆方向の指標のため、確率スコアとしては
    反転が必要 (詳細はモジュール冒頭docstring参照)。
    """
    pred_d = PredictorPredictions(
        name="案D",
        prob_taiou_success=df["prob_taiou_success_oof"].values,
        net_ojama_after_pred=df["net_ojama_after_oof_pred"].values,
    )
    pred_sim = PredictorPredictions(
        name="修正シミュ",
        prob_taiou_success=1.0 - df[SIM_SCORE_COL].values,
        net_ojama_after_pred=df[SIM_SCORE_COL].values,
    )
    pred_stack = PredictorPredictions(
        name="併用(スタッキング)",
        prob_taiou_success=oof_proba_stack,
        net_ojama_after_pred=oof_pred_stack,
    )
    return [pred_d, pred_sim, pred_stack]


# =============================================================================
# メイン
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する (main を50行以内に保つための分割)。"""
    parser = argparse.ArgumentParser(description="#24 打ち合い計測器 三つ巴比較 駆動スクリプト")
    parser.add_argument("--aug-csv", type=Path, default=DEFAULT_AUG_CSV)
    parser.add_argument("--model-d-dir", type=Path, default=DEFAULT_MODEL_D_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--limit", type=int, default=None,
                         help="先頭N行だけで動作確認する (本走行では未指定)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_exchange_triple_comparison] aug={args.aug_csv} model_d_dir={args.model_d_dir}")
    print("\n=== 1. 入力読込 ===")
    aug_df = load_exchange_labels(str(args.aug_csv))
    if args.limit is not None:
        aug_df = aug_df.head(args.limit).reset_index(drop=True)
    oof_df = load_model_d_oof(args.model_d_dir)

    print("\n=== 2. 案D OOF との突合 ===")
    merged = align_aug_with_model_d(aug_df, oof_df)

    print("\n=== 3. sim_damage_score NaN 除外 + 符号診断 ===")
    merged = filter_nan_sim_rows(merged)
    _log_sign_diagnostics(merged)

    print("\n=== 4. スタッキング版 (併用) OOF 学習 ===")
    oof_proba_stack, oof_pred_stack, feature_names = build_stacking_oof_predictions(merged, args.n_folds)
    print(f"  スタッキング特徴量: {feature_names}")

    print("\n=== 5. 三つ巴比較ハーネス実行 ===")
    predictors = build_predictor_set(merged, oof_proba_stack, oof_pred_stack)
    compare_predictors(merged, predictors, out_dir)
    print(f"\n出力先: {out_dir}")
    print("=== 完了 ===")


if __name__ == "__main__":
    main()
