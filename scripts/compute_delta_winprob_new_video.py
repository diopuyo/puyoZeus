"""未知動画 (学習66本に含まれない) 向け ΔWinProb 計算 (RT単体モデル版)。

## 背景・設計判断 (正直な注記)
`scripts.compute_exchange_delta_winprob` の `--recompute` 経路
(`load_aug_with_stacking_predictions`) は「併用スタッキング」モデルを使うが、
これは 66動画の aug CSV + 案D の **OOF (out-of-fold) 予測** を突合する設計
であり、aug CSV に存在しない genuinely 新規の動画には原理的に使えない
(inner join で 0 行になる。スタッキングメタモデル自体も cross-val 学習
専用でpersist されたモデルが無い)。

そのため本スクリプトは、新規動画の推論には
**案D単体の RT (リアルタイム) 推論バンドル** (`src.exchange_predictor`、
`scripts/train_exchange_model_d.py --save-model` で全66動画で学習し
joblib 保存済み、66動画には依存しない self-contained モデル) を使う。
併用スタッキングとの比較で AUC/rho がやや劣ることは事前の三つ巴比較
(exchange_triple_comparison) で判明済みだが、genuinely 新規動画に対して
リーク無く適用できる唯一の既存資産のため採用する。

勝率モデル (`train_winprob_models`) 自体は 66動画の labeled_win CSV で
学習するが、これは「新規動画の盤面グリッドを評価する関数」であり
新規動画のデータを学習に使うわけではない (リークではない)。

## 使い方
    PYTHONPATH=. python -m scripts.compute_delta_winprob_new_video \\
        --labels-csv data/indicators_v2/exchange_labels_olRyxDGacbg_2026-08-03.csv \\
        --npz-dir data/indicators_v2/boards_lean_olRyxDGacbg_2026-08-03 \\
        --exchange-model-path data/models/exchange_model_d_rt_2026-08-02.joblib \\
        --out-csv data/verify/delta_winprob_olRyxDGacbg_2026-08-03/exchange_delta_winprob.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.chain import ChainSimulator
from src.exchange_predictor import ExchangeModelBundle, load_exchange_model, predict_exchange_event
from scripts.compute_exchange_delta_winprob import (
    DEFAULT_LABELED_WIN_CSV,
    _load_video_npz,
    compute_delta_winprob_for_event,
    reconstruct_event_board_pair,
    train_winprob_models,
)

# =============================================================================
# 定数定義
# =============================================================================

DEFAULT_EXCHANGE_MODEL_PATH = Path("data/models/exchange_model_d_rt_2026-08-02.joblib")


def _build_rt_features(row: "pd.Series", model: ExchangeModelBundle) -> dict:
    """label_exchange_outcome.py 出力行 1件から RT推論用 features dict を作る。"""
    features: dict = {"phase": row["phase"], "fire_side": row["fire_side"]}
    for prefix in ("fire_", "opp_", "diff_"):
        for base in model.indicator_bases:
            features[f"{prefix}{base}"] = float(row[f"{prefix}{base}"])
    return features


def compute_events_for_new_video(
    labels_csv: Path, npz_dir: Path, exchange_model_path: Path, labeled_win_csv: Path,
) -> pd.DataFrame:
    """未知動画のΔWinProbイベントCSV (既存66動画版と同一スキーマ) を計算する。"""
    print(f"[new_video] ラベルCSV読込: {labels_csv}")
    labels_df = pd.read_csv(labels_csv)
    print(f"[new_video] RTモデル読込: {exchange_model_path}")
    rt_model = load_exchange_model(exchange_model_path)
    print("[new_video] 勝率モデル学習 (66動画 labeled_win、対称化・修正2件反映済み)")
    winprob_models = train_winprob_models(labeled_win_csv)

    video_caches: dict[str, "object"] = {}
    sim = ChainSimulator()
    rows: list[dict] = []
    for _, row in labels_df.iterrows():
        video_id = str(row["video_id"])
        if video_id not in video_caches:
            video_caches[video_id] = _load_video_npz(video_id, npz_dir)
        cache = video_caches[video_id]
        out = dict(row)
        if cache is None:
            out["match_failed"] = True
            rows.append(out)
            continue
        pair = reconstruct_event_board_pair(cache, int(row["game_idx"]), float(row["t_sec"]), str(row["fire_side"]))
        if pair is None:
            out["match_failed"] = True
            rows.append(out)
            continue
        fire_board, opp_board = pair
        _prob_taiou, net_ojama_after_pred = predict_exchange_event(rt_model, _build_rt_features(row, rt_model))
        delta = compute_delta_winprob_for_event(
            fire_board, opp_board, str(row["phase"]), net_ojama_after_pred, winprob_models, sim,
        )
        out.update(
            match_failed=False,
            net_ojama_after_pred_rt=net_ojama_after_pred,
            winprob_before=delta.winprob_before,
            winprob_after=delta.winprob_after,
            delta_winprob=delta.delta_winprob,
            attacker_dead_after=delta.attacker_dead_after,
            opponent_dead_after=delta.opponent_dead_after,
        )
        rows.append(out)
    out_df = pd.DataFrame(rows)
    n_failed = int(out_df["match_failed"].sum())
    print(f"[new_video] 盤面突合失敗={n_failed}/{len(out_df)}行")
    return out_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="未知動画向けΔWinProb計算 (RT単体モデル版)")
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
