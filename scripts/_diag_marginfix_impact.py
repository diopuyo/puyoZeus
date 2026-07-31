"""marginタイム elapsed_sec バグ修正前後の exchange_labels.csv を比較する診断スクリプト。

修正前 (exchange_labels_pre_marginfix.csv) と修正後 (exchange_labels.csv) を
(video_id, game_idx, fire_side, t_sec) キーで突き合わせ、火力系列
(potential_fire_power / immediate_fire_power / honsen_output) の
game_idx==0 / game_idx>=1 別の変化量・膨張倍率を報告する。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PRE_CSV = Path("data/indicators_v2/exchange_labels_pre_marginfix.csv")
POST_CSV = Path("data/indicators_v2/exchange_labels.csv")

# 比較対象の火力系列(fire/opp/diff)
FIRE_COLS: list[str] = [
    "fire_potential_fire_power", "opp_potential_fire_power", "diff_potential_fire_power",
    "fire_immediate_fire_power", "opp_immediate_fire_power", "diff_immediate_fire_power",
    "fire_honsen_output", "opp_honsen_output", "diff_honsen_output",
]

# 純粋盤面量(marginと無関係のはず、不変チェック対象)
PURE_COLS: list[str] = [
    "fire_current_max_chain", "fire_death_margin", "fire_board_ojama_count",
    "fire_absorption_capacity", "fire_max_column_height",
]


def _report_group(pre: pd.DataFrame, post: pd.DataFrame, label: str) -> None:
    """1グループ(game_idx==0 / >=1)分の変化量サマリを出力する。"""
    print(f"\n=== {label} (n={len(pre)}) ===")
    for col in FIRE_COLS:
        pre_v = pre[col].values.astype(np.float64)
        post_v = post[col].values.astype(np.float64)
        diff = post_v - pre_v
        # 膨張倍率(0除算回避のためpre>0の行のみ)
        nonzero_mask = pre_v > 1e-9
        ratio = np.where(nonzero_mask, post_v / np.maximum(pre_v, 1e-9), np.nan)
        n_changed = int((np.abs(diff) > 1e-6).sum())
        print(
            f"  {col:32s} pre_mean={pre_v.mean():10.2f} post_mean={post_v.mean():10.2f} "
            f"diff_mean={diff.mean():+10.2f} diff_max_abs={np.abs(diff).max():10.2f} "
            f"changed_rows={n_changed}/{len(pre)} "
            f"mean_ratio(post/pre, pre>0のみ)={np.nanmean(ratio):.3f} "
            f"max_ratio={np.nanmax(ratio) if nonzero_mask.any() else float('nan'):.3f}"
        )


def main() -> None:
    """メイン処理: 修正前後CSVを読み込み突き合わせて差分を報告する。"""
    pre = pd.read_csv(PRE_CSV)
    post = pd.read_csv(POST_CSV)
    print(f"[INFO] pre  shape={pre.shape}")
    print(f"[INFO] post shape={post.shape}")

    key_cols = ["video_id", "game_idx", "fire_side", "t_sec"]
    pre_k = pre.set_index(key_cols, drop=False)
    post_k = post.set_index(key_cols, drop=False)
    common_idx = pre_k.index.intersection(post_k.index)
    print(f"[INFO] キー一致行数: {len(common_idx)} / pre={len(pre_k)} / post={len(post_k)}")

    pre_c = pre_k.loc[common_idx].sort_index()
    post_c = post_k.loc[common_idx].sort_index()

    # 純粋盤面量が完全不変か確認
    print("\n=== 純粋盤面量(marginと無関係) 不変性チェック ===")
    for col in PURE_COLS:
        max_abs_diff = float(np.abs(post_c[col].values - pre_c[col].values).max())
        print(f"  {col:32s} max_abs_diff={max_abs_diff:.6f} ({'不変' if max_abs_diff < 1e-6 else '変化あり!'})")

    mask0 = pre_c["game_idx"] == 0
    mask1 = pre_c["game_idx"] >= 1
    _report_group(pre_c[mask0], post_c[mask0], "game_idx==0")
    _report_group(pre_c[mask1], post_c[mask1], "game_idx>=1")


if __name__ == "__main__":
    main()
