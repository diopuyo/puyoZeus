"""#24 打ち合い計測器「併用スタッキング」版 推論バンドル学習+永続化 (2026-08-03)。

## 背景
`scripts.compute_delta_winprob_new_video` は当初「案D単体」RT バンドル
(`data/models/exchange_model_d_rt_2026-08-02.joblib`) を使っていたが、
user レビュー (match_01/02/04) で taiou_success 予測の大外れが確認され、
「未知動画のフォールバックとして案D単体を使う」設計を廃止し、三つ巴比較で
最良だった **併用スタッキング** (案D特徴量41 + sim_* 3列=44特徴量) を
新規動画にも配備できるようにする。

## 設計方針 (既存資産の再利用・再実装禁止)
特徴量構築・学習ハイパラ・最終モデルfit・保存は全て
`scripts.train_exchange_model_d` の既存関数を再利用する:
    - `load_exchange_labels` / `get_indicator_base_names` / `build_feature_matrix`
      (`extra_feature_cols=SIM_FEATURE_COLS` で44特徴量に拡張、既存 optional 引数)
    - `fit_final_models` (全データ最終fit、run_exchange_triple_comparison の
      OOF学習とは別物。OOFはあくまで評価用、本番配備用モデルはここで作る)
    - `save_model_bundle` (2026-08-03 追加の `sim_feature_cols` optional引数で
      併用スタッキング形式のバンドルを保存)
`scripts.run_exchange_triple_comparison.SIM_FEATURE_COLS` / `filter_nan_sim_rows`
もそのまま再利用する (sim_* 列名・NaN除外ロジックを重複させない)。

## 使い方
    PYTHONPATH=. python -m scripts.train_exchange_stacking_rt \\
        --aug-csv data/indicators_v2/exchange_labels_regen_step3_aug_2026-08-02.csv \\
        --save-model data/models/exchange_stacking_rt_2026-08-03.joblib
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from scripts.run_exchange_triple_comparison import SIM_FEATURE_COLS, filter_nan_sim_rows
from scripts.train_exchange_model_d import (
    build_feature_matrix,
    fit_final_models,
    get_indicator_base_names,
    load_exchange_labels,
    save_model_bundle,
)

DEFAULT_AUG_CSV = Path("data/indicators_v2/exchange_labels_regen_step3_aug_2026-08-02.csv")
DEFAULT_SAVE_PATH = Path("data/models/exchange_stacking_rt_2026-08-03.joblib")


def train_and_save_stacking_bundle(
    aug_csv: Path, save_path: Path, model_date: "str | None" = None,
) -> None:
    """aug CSV (sim_*列付き) 全件で併用スタッキング版の最終モデルを学習・保存する。"""
    print(f"[stacking_rt] aug CSV読込: {aug_csv}")
    df = load_exchange_labels(str(aug_csv))
    df = filter_nan_sim_rows(df)

    indicator_bases = get_indicator_base_names(df)
    print(f"[stacking_rt] 指標base数={len(indicator_bases)} + sim_*{len(SIM_FEATURE_COLS)}列")
    X, feature_names = build_feature_matrix(df, indicator_bases, extra_feature_cols=list(SIM_FEATURE_COLS))
    y_cls = df["taiou_success"].astype(int).values
    y_reg = df["net_ojama_after"].astype(float).values
    print(f"[stacking_rt] 特徴量数={len(feature_names)} サンプル数={len(df)}")

    cls_model, reg_model = fit_final_models(X, y_cls, y_reg)
    save_model_bundle(
        cls_model, reg_model, indicator_bases, feature_names,
        str(aug_csv), model_date or datetime.date.today().isoformat(), len(df), save_path,
        sim_feature_cols=tuple(SIM_FEATURE_COLS),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="併用スタッキング版 推論バンドル学習+永続化")
    parser.add_argument("--aug-csv", type=Path, default=DEFAULT_AUG_CSV)
    parser.add_argument("--save-model", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--model-date", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_and_save_stacking_bundle(args.aug_csv, args.save_model, args.model_date)


if __name__ == "__main__":
    main()
