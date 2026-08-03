"""未知動画 (学習66本に含まれない) 向け ΔWinProb 計算 (併用スタッキング版)。

## 経緯 (2026-08-03 修正: 案D単体フォールバックを廃止)
当初は「案D単体」RT バンドルを使っていたが、user レビュー (match_01の
2942.9s 2P6連鎖、match_02の2984.9s 2P5連鎖、match_04の3097.3s 2P5連鎖) で
taiou_success 予測が大外れ (7.3%〜34.8%) することが確認され、三つ巴比較で
最良だった **併用スタッキング** (案D特徴量41 + sim_* 3列=44特徴量) に
切り替える。

## 設計方針 (既存資産の再利用・再実装禁止)
- 併用スタッキングモデル自体は `scripts.train_exchange_stacking_rt` が
  66動画 aug CSV 全件で学習・joblib 保存済みの self-contained バンドル
  (66動画には推論時点で依存しない、`src.exchange_predictor` 経由で読む)。
- sim_* 3列 (sim_k_hands/sim_expected_counter_ojama/sim_damage_score) は
  66動画では事前一括計算済み (aug CSV) だが、新規動画にはその事前計算が
  無いため **発火イベント毎にその場で計算** する
  (`scripts.augment_exchange_labels_with_sim._compute_sim_columns_for_row`
  をそのまま import して使う、コピペ再実装しない。1イベント約1秒だが
  動画モード=数十件規模のため許容)。

勝率モデル (`train_winprob_models`) 自体は 66動画の labeled_win CSV で
学習するが、これは「新規動画の盤面グリッドを評価する関数」であり
新規動画のデータを学習に使うわけではない (リークではない)。

## 使い方
    PYTHONPATH=. python -m scripts.compute_delta_winprob_new_video \\
        --labels-csv data/indicators_v2/exchange_labels_olRyxDGacbg_2026-08-03.csv \\
        --npz-dir data/indicators_v2/boards_lean_olRyxDGacbg_2026-08-03 \\
        --exchange-model-path data/models/exchange_stacking_rt_2026-08-03.joblib \\
        --out-csv data/verify/delta_winprob_olRyxDGacbg_2026-08-03/exchange_delta_winprob.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.chain import ChainSimulator
from src.exchange_predictor import ExchangeModelBundle, load_exchange_model, predict_exchange_event
from scripts.augment_exchange_labels_with_sim import _VideoCache, _compute_sim_columns_for_row, _load_video_cache
from scripts.compute_exchange_delta_winprob import (
    DEFAULT_LABELED_WIN_CSV,
    _load_video_npz,
    apply_mutual_exchange_adjustment,
    compute_delta_winprob_for_event,
    reconstruct_event_board_pair,
    train_winprob_models,
)

# =============================================================================
# 定数定義
# =============================================================================

DEFAULT_EXCHANGE_MODEL_PATH = Path("data/models/exchange_stacking_rt_2026-08-03.joblib")

# augment_exchange_labels_with_sim.py の --mode 既定値と揃える
# (aug CSV = exchange_labels_regen_step3_aug_2026-08-02.csv 生成時の値に
# ついて明示ログが残っておらず、CLI既定 "precise" が使われた前提の仮定。
# 万一 "fast" だった場合は sim_damage_score の分布がわずかに変わりうるが
# k_hands/expected_counter_ojama の計算自体は mode に依存しない箇所が主)。
SIM_MODE: str = "precise"


def _build_features(row: "pd.Series", model: ExchangeModelBundle,
                    sim_values: tuple[float, float, float]) -> dict:
    """label_exchange_outcome.py 出力行 1件 + その場計算した sim_* から features dict を作る。"""
    features: dict = {"phase": row["phase"], "fire_side": row["fire_side"]}
    for prefix in ("fire_", "opp_", "diff_"):
        for base in model.indicator_bases:
            features[f"{prefix}{base}"] = float(row[f"{prefix}{base}"])
    for col, val in zip(model.sim_feature_cols, sim_values):
        features[col] = val
    return features


def compute_events_for_new_video(
    labels_csv: Path, npz_dir: Path, exchange_model_path: Path, labeled_win_csv: Path,
) -> pd.DataFrame:
    """未知動画のΔWinProbイベントCSV (既存66動画版と同一スキーマ) を計算する。"""
    print(f"[new_video] ラベルCSV読込: {labels_csv}")
    labels_df = pd.read_csv(labels_csv)
    print(f"[new_video] 併用スタッキングモデル読込: {exchange_model_path}")
    stack_model = load_exchange_model(exchange_model_path)
    if not stack_model.sim_feature_cols:
        print("[new_video] 警告: 読み込んだバンドルに sim_feature_cols が無い"
              " (案D単体バンドルの可能性、sim_*計算はスキップされます)")
    print("[new_video] 勝率モデル学習 (66動画 labeled_win、対称化・修正2件反映済み)")
    winprob_models = train_winprob_models(labeled_win_csv)

    npz_caches: dict[str, "object"] = {}
    sim_caches: dict[str, "_VideoCache | None"] = {}
    sim = ChainSimulator()

    # --- Pass 1: 盤面突合 + 併用スタッキング予測 (相打ち相殺前の一次値) ---
    rows: list[dict] = []
    for _, row in labels_df.iterrows():
        video_id = str(row["video_id"])
        if video_id not in npz_caches:
            npz_caches[video_id] = _load_video_npz(video_id, npz_dir)
            sim_caches[video_id] = _load_video_cache(video_id, npz_dir)
        cache = npz_caches[video_id]
        sim_cache = sim_caches[video_id]
        out = dict(row)
        if cache is None or sim_cache is None:
            out["match_failed"] = True
            rows.append(out)
            continue
        pair = reconstruct_event_board_pair(cache, int(row["game_idx"]), float(row["t_sec"]), str(row["fire_side"]))
        if pair is None:
            out["match_failed"] = True
            rows.append(out)
            continue
        sim_k_hands, sim_expected_counter_ojama, sim_damage_score = _compute_sim_columns_for_row(
            row, sim_cache, SIM_MODE)
        features = _build_features(
            row, stack_model, (sim_k_hands, sim_expected_counter_ojama, sim_damage_score))
        _prob_taiou, net_ojama_after_pred = predict_exchange_event(stack_model, features)
        out.update(
            match_failed=False,
            sim_k_hands=sim_k_hands,
            sim_expected_counter_ojama=sim_expected_counter_ojama,
            sim_damage_score=sim_damage_score,
            stack_prob_taiou_success=_prob_taiou,
            stack_net_ojama_after_pred=net_ojama_after_pred,
        )
        rows.append(out)
    out_df = pd.DataFrame(rows)

    # --- Pass 2: 相打ち(欠陥D)検出・実測net_ojama_after相殺で上書き ---
    # (動画1本前提のスクリプトのため video_id 単位のループは不要、対象動画の
    # cache をそのまま使う。match_failed 行は突合済み盤面が無いため対象外)
    valid_mask = ~out_df["match_failed"]
    for video_id, cache in npz_caches.items():
        if cache is None:
            continue
        sub_mask = valid_mask & (out_df["video_id"] == video_id)
        if not sub_mask.any():
            continue
        adjusted_sub = apply_mutual_exchange_adjustment(out_df.loc[sub_mask], cache)
        out_df.loc[sub_mask, ["stack_net_ojama_after_pred", "is_mutual_exchange", "mutual_partner_t_sec"]] = (
            adjusted_sub[["stack_net_ojama_after_pred", "is_mutual_exchange", "mutual_partner_t_sec"]]
        )
    if "is_mutual_exchange" not in out_df.columns:
        out_df["is_mutual_exchange"] = False
        out_df["mutual_partner_t_sec"] = float("nan")
    n_mutual = int(out_df.loc[valid_mask, "is_mutual_exchange"].sum())
    print(f"[new_video] 相打ち(欠陥D)検出: {n_mutual}/{int(valid_mask.sum())}行")

    # --- Pass 3: (相打ち相殺後の) net_ojama_after_pred で ΔWinProb を計算 ---
    delta_rows: list[dict] = []
    for _, row in out_df.iterrows():
        if row["match_failed"]:
            delta_rows.append({"winprob_before": float("nan"), "winprob_after": float("nan"),
                               "delta_winprob": float("nan"), "attacker_dead_after": False,
                               "opponent_dead_after": False})
            continue
        cache = npz_caches[str(row["video_id"])]
        pair = reconstruct_event_board_pair(cache, int(row["game_idx"]), float(row["t_sec"]), str(row["fire_side"]))
        fire_board, opp_board = pair
        delta = compute_delta_winprob_for_event(
            fire_board, opp_board, str(row["phase"]), float(row["stack_net_ojama_after_pred"]),
            winprob_models, sim,
        )
        delta_rows.append({
            "winprob_before": delta.winprob_before, "winprob_after": delta.winprob_after,
            "delta_winprob": delta.delta_winprob, "attacker_dead_after": delta.attacker_dead_after,
            "opponent_dead_after": delta.opponent_dead_after,
        })
    delta_df = pd.DataFrame(delta_rows, index=out_df.index)
    out_df = pd.concat([out_df, delta_df], axis=1)

    n_failed = int(out_df["match_failed"].sum())
    print(f"[new_video] 盤面突合失敗={n_failed}/{len(out_df)}行")
    return out_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="未知動画向けΔWinProb計算 (併用スタッキング版)")
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--exchange-model-path", type=Path, default=DEFAULT_EXCHANGE_MODEL_PATH)
    parser.add_argument("--labeled-win-csv", type=Path, default=DEFAULT_LABELED_WIN_CSV)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = compute_events_for_new_video(
        args.labels_csv, args.npz_dir, args.exchange_model_path, args.labeled_win_csv)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[done] CSV保存: {args.out_csv} ({len(df)}行)")


if __name__ == "__main__":
    main()
