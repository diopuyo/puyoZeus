"""#24 打ち合い計測器 Step5「乖離上位イベント選定」(2026-08-02)。

三つ巴比較 (案D/修正シミュ/併用スタッキング) の本走行が完了し、併用スタッキングが
全位相で有意勝ちと判明した。採否確定権は user にあり数値だけで決めない規律
(feedback_viz_eval_required) のため、案Dと修正シミュの判断が最も割れたイベントを
実ゲーム画面で提示する資料 (review_sheet.md) の元になるイベント一覧を選ぶ。

既存資産の組み合わせのみで構成する (再実装禁止):
    - scripts/run_exchange_triple_comparison.py: 突合 (align_aug_with_model_d)・
      NaN除外 (filter_nan_sim_rows)・スタッキングOOF学習 (build_stacking_oof_predictions)。
    - scripts/train_exchange_model_d.py: aug CSV読込 (load_exchange_labels)。
    - scripts/exchange_meter_eval_harness.py: EXCHANGE_PHASES。

## 選定方法
    1. 案D の net_ojama_after 予測 と sim_damage_score をそれぞれ全体内
       パーセンタイル順位に変換し、|順位差| を「乖離度」とする。
    2. 主系列: 乖離度が大きい順に位相別 (序/中/終) 各4件 = 12件。
    3. 副系列: 併用スタッキングの net_ojama_after 予測が実測 (net_ojama_after)
       から最も外れた (順位ベース) 4件。run_exchange_triple_comparison.py は
       スタッキングOOFをファイル保存していないため、本スクリプト内で同一の
       fold分割・ハイパラで再学習する (再学習は同スクリプトの実測で数分程度、
       コピペ再実装はしない)。
    4. 動画DL節約の制約: 選定は合計3動画以内に収まるよう、乖離度上位が
       集中している動画を優先する (video_id ごとの上位N件乖離度合計でスコア化)。
       制約を満たすため順位を多少飛ばす場合は件数を必ずログする。

## 出力
    選定イベント一覧CSV (video_id, game_idx, t_sec, fire_side, phase,
    案D予測, sim予測, sim_k_hands, sim_expected_counter_ojama, 実測net_ojama_after,
    taiou_success, survived, 乖離度 + 選定系列/機械判定などの補助列)。

## 使い方
    PYTHONPATH=. python -m scripts.select_exchange_divergence_events \\
        --aug-csv data/indicators_v2/exchange_labels_regen_step0_aug_2026-08-01.csv \\
        --model-d-dir data/verify/exchange_model_d_regen_2026-08-02 \\
        --out-csv data/verify/exchange_divergence_review_2026-08-02/selected_events.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.exchange_meter_eval_harness import EXCHANGE_PHASES
from scripts.run_exchange_triple_comparison import (
    DEFAULT_AUG_CSV,
    DEFAULT_MODEL_D_DIR,
    SIM_SCORE_COL,
    align_aug_with_model_d,
    build_stacking_oof_predictions,
    filter_nan_sim_rows,
    load_model_d_oof,
)
from scripts.train_exchange_model_d import N_FOLDS, load_exchange_labels

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

DEFAULT_OUT_CSV = Path("data/verify/exchange_divergence_review_2026-08-02/selected_events.csv")

# 主系列: 位相別に選ぶ件数 (序/中/終 各4件 = 12件)
N_PRIMARY_PER_PHASE: int = 4

# 副系列: 併用スタッキングの外れ値件数
N_SECONDARY: int = 4

# 動画DL節約の制約 (合計何動画以内に収めるか)
MAX_VIDEOS: int = 3

# 動画スコアリングに使う「上位何件の乖離度合計か」 (主系列12+副系列4の総予算)
VIDEO_SCORE_BUDGET: int = N_PRIMARY_PER_PHASE * len(EXCHANGE_PHASES) + N_SECONDARY

# 案D予測・実測・sim予測列名 (align_aug_with_model_d の出力に含まれる列)
MODEL_D_PRED_COL: str = "net_ojama_after_oof_pred"
ACTUAL_NET_OJAMA_COL: str = "net_ojama_after"

# 併用スタッキングOOF予測を格納する列名 (本スクリプトが新規に追加する列)
STACK_PRED_COL: str = "stack_pred_net_ojama_after"

# イベント突合キー (run_exchange_triple_comparison.MERGE_KEYS と同一)
EVENT_KEYS: tuple[str, ...] = ("video_id", "game_idx", "t_sec", "fire_side")


# =============================================================================
# 1. 入力読込・突合 (既存資産の再利用のみ)
# =============================================================================

def load_merged_df(aug_csv: Path, model_d_dir: Path) -> pd.DataFrame:
    """aug CSV + 案D OOF を突合・NaN除外済みの DataFrame にして返す。"""
    aug_df = load_exchange_labels(str(aug_csv))
    oof_df = load_model_d_oof(model_d_dir)
    merged = align_aug_with_model_d(aug_df, oof_df)
    return filter_nan_sim_rows(merged)


# =============================================================================
# 2. 乖離度計算 (全体内パーセンタイル順位の差)
# =============================================================================

def add_rank_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """案D予測・sim予測・実測 それぞれの全体内パーセンタイル順位 + 乖離度を付与する。

    乖離度 = |案Dのパーセンタイル順位 - simのパーセンタイル順位| (全体内、位相を
    問わず算出。位相別選定は後段で行う)。
    """
    out = df.copy()
    out["rank_pct_model_d"] = out[MODEL_D_PRED_COL].rank(pct=True)
    out["rank_pct_sim"] = out[SIM_SCORE_COL].rank(pct=True)
    out["rank_pct_actual"] = out[ACTUAL_NET_OJAMA_COL].rank(pct=True)
    out["divergence_rank_pct"] = (out["rank_pct_model_d"] - out["rank_pct_sim"]).abs()
    return out


def judge_closer_to_actual(df: pd.DataFrame) -> pd.Series:
    """案D予測とsim予測、どちらが実測の順位に近いか (順位ベース参考判定)。

    値の単位が異なる2予測 (案Dはnet_ojama_afterと同じ単位、simは0〜1スコア) を
    公平に比べるため、生値の差でなく全体内パーセンタイル順位の差で比較する。
    """
    diff_model_d = (df["rank_pct_model_d"] - df["rank_pct_actual"]).abs()
    diff_sim = (df["rank_pct_sim"] - df["rank_pct_actual"]).abs()
    return np.where(diff_model_d <= diff_sim, "案D", "修正シミュ")


# =============================================================================
# 3. 動画選定 (乖離度上位が集中している動画を最大3本まで優先)
# =============================================================================

def score_videos_by_divergence_concentration(
    df: pd.DataFrame, top_n: int = VIDEO_SCORE_BUDGET,
) -> pd.Series:
    """動画ごとに乖離度上位 top_n 件の合計でスコア化する (集中度が高い動画ほど高スコア)。"""
    def _video_score(group: pd.DataFrame) -> float:
        return float(group["divergence_rank_pct"].nlargest(top_n).sum())
    return df.groupby("video_id").apply(_video_score).sort_values(ascending=False)


def select_allowed_videos(df: pd.DataFrame, max_videos: int = MAX_VIDEOS) -> list[str]:
    """乖離度集中スコア上位から最大 max_videos 本の動画IDを選ぶ。"""
    scores = score_videos_by_divergence_concentration(df)
    return list(scores.index[:max_videos])


def count_skipped_for_video_constraint(
    df_sorted_desc: pd.DataFrame, allowed_videos: list[str], n_needed: int,
) -> int:
    """動画制約により乖離度上位から何件飛ばしたかを数える (silent skip 禁止)。

    df_sorted_desc は乖離度 (または対象メトリック) 降順に並んでいる前提。
    allowed_videos に無い動画の行を「飛ばした」件数として数える
    (n_needed 件確保できた時点で数え終える)。
    """
    skipped = 0
    collected = 0
    for video_id in df_sorted_desc["video_id"]:
        if collected >= n_needed:
            break
        if video_id in allowed_videos:
            collected += 1
        else:
            skipped += 1
    return skipped


# =============================================================================
# 4. 主系列選定 (位相別 乖離度上位)
# =============================================================================

def select_primary_events(
    df: pd.DataFrame, allowed_videos: list[str], n_per_phase: int = N_PRIMARY_PER_PHASE,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """位相別 (序/中/終) に乖離度上位 n_per_phase 件を allowed_videos 内から選ぶ。"""
    rows: list[pd.DataFrame] = []
    skipped_by_phase: dict[str, int] = {}
    for phase in EXCHANGE_PHASES:
        phase_df = df[df["phase"] == phase].sort_values("divergence_rank_pct", ascending=False)
        skipped_by_phase[phase] = count_skipped_for_video_constraint(phase_df, allowed_videos, n_per_phase)
        selected = phase_df[phase_df["video_id"].isin(allowed_videos)].head(n_per_phase).copy()
        selected["selection_reason"] = [f"{phase}_乖離度{i + 1}位" for i in range(len(selected))]
        rows.append(selected)
    primary = pd.concat(rows, ignore_index=True) if rows else df.iloc[0:0].copy()
    primary["selection_series"] = "主系列"
    return primary, skipped_by_phase


# =============================================================================
# 5. 副系列選定 (併用スタッキングの外れ値)
# =============================================================================

def attach_stacking_predictions(df: pd.DataFrame, n_folds: int = N_FOLDS) -> pd.DataFrame:
    """併用スタッキングの net_ojama_after OOF予測を計算し列として付与する。

    train_exchange_model_d/run_exchange_triple_comparison の関数を再利用する
    (コピペ再実装しない)。
    """
    _oof_proba_stack, oof_pred_stack, feature_names = build_stacking_oof_predictions(df, n_folds)
    print(f"  併用スタッキング特徴量: {feature_names}")
    out = df.copy()
    out[STACK_PRED_COL] = oof_pred_stack
    return out


def select_secondary_events(
    df_with_stack: pd.DataFrame,
    allowed_videos: list[str],
    exclude_keys: set[tuple],
    n_secondary: int = N_SECONDARY,
) -> tuple[pd.DataFrame, int]:
    """併用スタッキングの net_ojama_after 予測が実測から最も外れた n_secondary 件を選ぶ

    (順位ベース: パーセンタイル順位の差が大きい順。主系列と重複するイベントは除外する)。
    """
    ranked = df_with_stack.copy()
    ranked["rank_pct_stack"] = ranked[STACK_PRED_COL].rank(pct=True)
    ranked["stack_residual_rank_pct"] = (ranked["rank_pct_stack"] - ranked["rank_pct_actual"]).abs()
    ranked = ranked.sort_values("stack_residual_rank_pct", ascending=False)

    is_excluded = ranked[list(EVENT_KEYS)].apply(tuple, axis=1).isin(exclude_keys)
    candidates = ranked[~is_excluded]
    skipped = count_skipped_for_video_constraint(candidates, allowed_videos, n_secondary)
    selected = candidates[candidates["video_id"].isin(allowed_videos)].head(n_secondary).copy()
    selected["selection_reason"] = [f"併用スタッキング外れ値{i + 1}位" for i in range(len(selected))]
    selected["selection_series"] = "副系列"
    return selected, skipped


# =============================================================================
# 6. 出力テーブル組み立て
# =============================================================================

OUTPUT_COLUMNS: tuple[str, ...] = (
    "video_id", "game_idx", "t_sec", "fire_side", "phase", "approx_fire_chains",
    MODEL_D_PRED_COL, SIM_SCORE_COL, "sim_k_hands", "sim_expected_counter_ojama",
    ACTUAL_NET_OJAMA_COL, "taiou_success", "survived", "divergence_rank_pct",
    STACK_PRED_COL, "closer_to_actual_rank_based", "selection_series", "selection_reason",
)


def build_output_table(primary_df: pd.DataFrame, secondary_df: pd.DataFrame) -> pd.DataFrame:
    """主系列+副系列を結合し、機械判定列を付与して出力列に整形する。"""
    combined = pd.concat([primary_df, secondary_df], ignore_index=True)
    combined["closer_to_actual_rank_based"] = judge_closer_to_actual(combined)
    return combined[list(OUTPUT_COLUMNS)]


# =============================================================================
# メイン
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する (main を50行以内に保つための分割)。"""
    parser = argparse.ArgumentParser(description="#24 打ち合い計測器 Step5 乖離上位イベント選定")
    parser.add_argument("--aug-csv", type=Path, default=DEFAULT_AUG_CSV)
    parser.add_argument("--model-d-dir", type=Path, default=DEFAULT_MODEL_D_DIR)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[select_exchange_divergence_events] aug={args.aug_csv} model_d_dir={args.model_d_dir}")

    print("\n=== 1. 入力読込・突合・NaN除外 ===")
    merged = load_merged_df(args.aug_csv, args.model_d_dir)

    print("\n=== 2. 乖離度計算 (全体内パーセンタイル順位) ===")
    merged = add_rank_percentiles(merged)

    print("\n=== 3. 動画選定 (乖離度集中スコア上位3本) ===")
    allowed_videos = select_allowed_videos(merged)
    print(f"  選定動画: {allowed_videos}")

    print("\n=== 4. 主系列選定 (位相別 乖離度上位) ===")
    primary_df, skipped_by_phase = select_primary_events(merged, allowed_videos)
    for phase, n_skipped in skipped_by_phase.items():
        print(f"  位相={phase}: 選定{int((primary_df['phase'] == phase).sum())}件"
              f"  動画制約で飛ばした件数={n_skipped}")

    print("\n=== 5. 副系列選定 (併用スタッキング外れ値) ===")
    merged_with_stack = attach_stacking_predictions(merged, args.n_folds)
    exclude_keys = set(primary_df[list(EVENT_KEYS)].apply(tuple, axis=1))
    secondary_df, n_skipped_secondary = select_secondary_events(
        merged_with_stack, allowed_videos, exclude_keys,
    )
    print(f"  副系列選定{len(secondary_df)}件  動画制約/主系列重複で飛ばした件数={n_skipped_secondary}")

    print("\n=== 6. 出力 ===")
    # 副系列にも rank_pct_* 列を揃えるため、主系列側も merged_with_stack 由来に統一する。
    primary_with_stack = merged_with_stack.merge(
        primary_df[list(EVENT_KEYS) + ["selection_reason", "selection_series"]],
        on=list(EVENT_KEYS), how="inner",
    )
    output = build_output_table(primary_with_stack, secondary_df)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out_csv, index=False)
    print(f"  選定イベント合計{len(output)}件 -> {args.out_csv}")
    print(f"  使用動画: {sorted(output['video_id'].unique())}")
    print("=== 完了 ===")


if __name__ == "__main__":
    main()
