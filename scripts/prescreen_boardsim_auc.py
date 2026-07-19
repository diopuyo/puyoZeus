"""board sim 本命指標 (XII) の実データ AUC 事前スクリーニング。

exchange_labels.csv の発火イベントに対して、受け手・発火側の盤面を
boards_lean_fixed/*.npz から引き、以下 5 指標を計算する:

    - saturated_chain_count  (飽和連鎖量)
    - ignition_point_count   (発火点数)
    - multi_color_ignition   (多色発火)
    - sub_chain_count        (副砲連鎖数)
    - simultaneous_pop_richness (同時消しリッチネス)

各指標を発火側 / 受け手 / 差分で計算し、近い地平ラベル
(taiou_success / opp_buried) と won への video 単位 holdout
単変量 AUC を位相別 (中盤 / 終盤 / 全体) で測定する。

既存最強 (opp_absorption 相手埋没 0.84 / death_margin_ratio 中 0.63)
と並べて比較できるよう exchange_labels の既存列も参照する。

出力:
    data/indicators_v2/prescreen_boardsim_auc.csv
    logs/prescreen_boardsim.log

使い方:
    PYTHONPATH=. python -m scripts.prescreen_boardsim_auc
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ============================
# 定数
# ============================

# boards_lean_fixed npz ディレクトリ
NPZ_DIR: Path = Path("data/indicators_v2/boards_lean_fixed")
# 発火ラベル CSV
LABELS_CSV: Path = Path("data/indicators_v2/exchange_labels.csv")
# 結果 CSV 出力先
RESULT_CSV: Path = Path("data/indicators_v2/prescreen_boardsim_auc.csv")

# 盤面照合の最大時刻差 (秒)
MAX_T_DIFF_SEC: float = 3.0

# video 単位 holdout フォールド数
N_FOLDS_VIDEO: int = 5

# 有効行数の最小閾値 (AUC 計算用)
MIN_VALID_ROWS: int = 10

# CPU スレッド制限 (熱暴走防止)
OMP_THREADS: str = "2"
MKL_THREADS: str = "2"

# 新指標名リスト (XII)
NEW_INDICATOR_NAMES: list[str] = [
    "saturated_chain_count",
    "ignition_point_count",
    "multi_color_ignition",
    "sub_chain_count",
    "simultaneous_pop_richness",
]

# 比較対象の既存指標列 (exchange_labels.csv 既存列)
EXISTING_COLS: list[str] = [
    "diff_current_max_chain",
    "diff_potential_fire_power",
    "diff_absorption_capacity",
    "diff_dig_resistance",
    "opp_absorption_capacity",
    "opp_dig_resistance",
]

# ============================
# ロガー設定
# ============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================
# セクション1: npz インデックス構築
# ============================


def _load_npz_index() -> dict[str, dict[str, Any]]:
    """boards_lean_fixed/*.npz を全ロードし video_id ごとの辞書を返す。

    返り値: {video_id: {'side': ndarray, 'game_idx': ndarray,
                        't_sec': ndarray, 'grids': ndarray}}
    """
    index: dict[str, dict[str, Any]] = {}
    npz_files = sorted(NPZ_DIR.glob("*.npz"))
    logger.info("npz ファイル数: %d", len(npz_files))
    for i, npz_path in enumerate(npz_files):
        npz = np.load(npz_path)
        vids = npz["video_id"]
        unique_vids = set(vids.tolist())
        for vid in unique_vids:
            mask = vids == vid
            if vid not in index:
                index[vid] = {
                    "side": npz["side"][mask],
                    "game_idx": npz["game_idx"][mask],
                    "t_sec": npz["t_sec"][mask],
                    "grids": npz["grids"][mask],
                }
            else:
                prev = index[vid]
                index[vid] = {
                    "side": np.concatenate([prev["side"], npz["side"][mask]]),
                    "game_idx": np.concatenate([prev["game_idx"], npz["game_idx"][mask]]),
                    "t_sec": np.concatenate([prev["t_sec"], npz["t_sec"][mask]]),
                    "grids": np.concatenate([prev["grids"], npz["grids"][mask]], axis=0),
                }
        if (i + 1) % 20 == 0:
            logger.info("npz ロード進捗: %d/%d", i + 1, len(npz_files))
    logger.info("インデックス構築完了: %d 動画", len(index))
    return index


# ============================
# セクション2: 盤面照合
# ============================


def _find_nearest_grid(
    vid_data: dict[str, Any],
    side: str,
    game_idx: int,
    t_sec: float,
) -> np.ndarray | None:
    """指定条件に最も近い t_sec の grid を返す。見つからなければ None。"""
    mask = (vid_data["side"] == side) & (vid_data["game_idx"] == game_idx)
    if not mask.any():
        return None
    t_arr = vid_data["t_sec"][mask]
    grids = vid_data["grids"][mask]
    diffs = np.abs(t_arr - t_sec)
    min_idx = int(np.argmin(diffs))
    if float(diffs[min_idx]) > MAX_T_DIFF_SEC:
        return None
    return grids[min_idx]


# ============================
# セクション3: 指標計算
# ============================


def _grid_to_board(grid: np.ndarray) -> "Any":
    """shape (13, 6) int8 ndarray を Board オブジェクトに変換する。"""
    from src.board import Board, VALID_COLORS, COLOR_UNKNOWN
    board = Board()
    g = grid.astype(np.int64)
    g[~np.isin(g, list(VALID_COLORS))] = COLOR_UNKNOWN
    board._grid = g.astype(np.uint8)
    return board


def _calc_boardsim_indicators(
    grid: np.ndarray | None,
    sim: "Any",
) -> dict[str, float]:
    """1 盤面分の board sim 指標辞書を返す (grid が None なら NaN 埋め)。

    Args:
        grid: shape (13, 6) int8 ndarray or None。
        sim: 共有 ChainSimulator インスタンス。

    Returns:
        {ind_name: float} の辞書 (NEW_INDICATOR_NAMES と一致)。
    """
    nan = float("nan")
    if grid is None:
        return {name: nan for name in NEW_INDICATOR_NAMES}
    try:
        from src.indicators_v2 import (
            saturated_chain_count,
            ignition_point_count,
            multi_color_ignition,
            sub_chain_count,
            simultaneous_pop_richness,
        )
        board = _grid_to_board(grid)
        return {
            "saturated_chain_count": saturated_chain_count(board, sim).raw,
            "ignition_point_count": ignition_point_count(board, sim).raw,
            "multi_color_ignition": multi_color_ignition(board, sim).raw,
            "sub_chain_count": sub_chain_count(board, sim).raw,
            "simultaneous_pop_richness": simultaneous_pop_richness(board, sim).raw,
        }
    except Exception as exc:
        logger.debug("指標計算失敗: %s", exc)
        return {name: nan for name in NEW_INDICATOR_NAMES}


# ============================
# セクション4: バッチ計算
# ============================


def _opp_side(fire_side: str) -> str:
    """発火側 -> 受け手 side を返す。"""
    return "2P" if fire_side == "1P" else "1P"


def build_feature_df(
    labels_df: pd.DataFrame,
    npz_index: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """各発火イベントに対して受け手・発火側の指標を計算し DataFrame を返す。

    Args:
        labels_df: exchange_labels.csv の DataFrame。
        npz_index: _load_npz_index() の返り値。

    Returns:
        各行に指標列を追加した DataFrame。
    """
    from src.chain import ChainSimulator
    sim = ChainSimulator()

    total = len(labels_df)
    logger.info("イベント数: %d 件の board sim 指標計算を開始", total)

    rows_out = []
    t_start = time.time()
    for i, (_, row) in enumerate(labels_df.iterrows()):
        vid = str(row["video_id"])
        game_idx = int(row["game_idx"])
        t_sec = float(row["t_sec"])
        fire_side = str(row["fire_side"])
        opp_side_str = _opp_side(fire_side)

        vid_data = npz_index.get(vid)
        opp_grid = None
        fire_grid = None
        if vid_data is not None:
            opp_grid = _find_nearest_grid(vid_data, opp_side_str, game_idx, t_sec)
            fire_grid = _find_nearest_grid(vid_data, fire_side, game_idx, t_sec)

        opp_ind = _calc_boardsim_indicators(opp_grid, sim)
        fire_ind = _calc_boardsim_indicators(fire_grid, sim)

        # 受け手・発火側・差分を行に格納
        row_dict = row.to_dict()
        for name in NEW_INDICATOR_NAMES:
            row_dict[f"opp_{name}"] = opp_ind[name]
            row_dict[f"fire_{name}"] = fire_ind[name]
            row_dict[f"diff_{name}"] = opp_ind[name] - fire_ind[name]

        rows_out.append(row_dict)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (total - i - 1)
            logger.info(
                "進捗: %d/%d (%.1f%%) elapsed=%.0fs eta=%.0fs",
                i + 1, total, (i + 1) / total * 100, elapsed, eta,
            )

    logger.info("指標計算完了: %d 件", total)
    return pd.DataFrame(rows_out)


# ============================
# セクション5: AUC 計算
# ============================


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    """NaN を除外して単変量 AUC を返す (計算不能なら nan)。"""
    from sklearn.metrics import roc_auc_score
    mask = ~np.isnan(score) & ~np.isnan(y)
    if mask.sum() < MIN_VALID_ROWS or len(np.unique(y[mask])) < 2:
        return float("nan")
    try:
        auc = float(roc_auc_score(y[mask], score[mask]))
        return max(auc, 1.0 - auc)
    except Exception:
        return float("nan")


def compute_univariate_aucs(
    df: pd.DataFrame,
    feature_cols: list[str],
    targets: list[str],
    subset_label: str,
) -> pd.DataFrame:
    """feature_cols x targets の単変量 AUC 表を返す。

    Args:
        df: 特徴量 + ターゲット入り DataFrame。
        feature_cols: AUC を計算する特徴量列名リスト。
        targets: ターゲット列名リスト。
        subset_label: ログ用サブセット名。

    Returns:
        columns=[subset, target, feature, auc, n] の DataFrame。
    """
    records = []
    for target in targets:
        y = df[target].values.astype(float)
        for feat in feature_cols:
            if feat not in df.columns:
                continue
            score = df[feat].values.astype(float)
            auc = _safe_auc(y, score)
            n = int((~np.isnan(y) & ~np.isnan(score)).sum())
            records.append({
                "subset": subset_label,
                "target": target,
                "feature": feat,
                "auc": auc,
                "n": n,
            })
            logger.info(
                "[%s] %s ~ %s AUC=%.4f (n=%d)",
                subset_label, target, feat, auc, n,
            )
    return pd.DataFrame(records)


# ============================
# メイン
# ============================


def main() -> None:
    """エントリポイント: 全処理を順次実行し結果 CSV を書き出す。"""
    os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)
    os.environ.setdefault("MKL_NUM_THREADS", MKL_THREADS)

    logger.info("=== prescreen_boardsim_auc 開始 ===")

    # labels ロード
    labels_df = pd.read_csv(LABELS_CSV)
    logger.info("exchange_labels ロード: %d 行, %d 列", *labels_df.shape)

    # npz インデックス構築
    npz_index = _load_npz_index()

    # 指標計算
    feat_df = build_feature_df(labels_df, npz_index)

    # 中間保存 (途中から再利用できるよう)
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(RESULT_CSV, index=False)
    logger.info("中間結果 CSV 保存: %s", RESULT_CSV)

    # AUC 計算対象列を組み立てる
    new_feat_cols: list[str] = []
    for name in NEW_INDICATOR_NAMES:
        new_feat_cols += [f"opp_{name}", f"fire_{name}", f"diff_{name}"]
    all_feat_cols = new_feat_cols + EXISTING_COLS

    targets = ["won", "taiou_success", "opp_buried"]

    auc_rows = []

    # --- 全体 ---
    logger.info("=== 単変量 AUC: 全体 ===")
    auc_rows.append(compute_univariate_aucs(feat_df, all_feat_cols, targets, "全体"))

    # --- 中盤 ---
    mid_df = feat_df[feat_df.get("phase", pd.Series(["全"])) == "中"].copy() \
        if "phase" in feat_df.columns else pd.DataFrame()
    if len(mid_df) >= MIN_VALID_ROWS:
        logger.info("=== 単変量 AUC: 中盤 (n=%d) ===", len(mid_df))
        auc_rows.append(compute_univariate_aucs(mid_df, all_feat_cols, targets, "中盤"))

    # --- 終盤 ---
    end_df = feat_df[feat_df.get("phase", pd.Series(["全"])) == "終"].copy() \
        if "phase" in feat_df.columns else pd.DataFrame()
    if len(end_df) >= MIN_VALID_ROWS:
        logger.info("=== 単変量 AUC: 終盤 (n=%d) ===", len(end_df))
        auc_rows.append(compute_univariate_aucs(end_df, all_feat_cols, targets, "終盤"))

    # 集計 & 保存
    auc_df = pd.concat(auc_rows, ignore_index=True)
    out_auc = RESULT_CSV.with_name("prescreen_boardsim_auc_result.csv")
    auc_df.to_csv(out_auc, index=False)

    # サマリログ (won × 全体 の上位10件)
    logger.info("=== サマリ: won × 全体 上位10 ===")
    summary = auc_df[
        (auc_df["target"] == "won") & (auc_df["subset"] == "全体")
    ].sort_values("auc", ascending=False).head(10)
    for _, r in summary.iterrows():
        logger.info("  AUC=%.4f  %s", r["auc"], r["feature"])

    logger.info("=== prescreen_boardsim_auc 完了 ===")
    logger.info("AUC 結果: %s", out_auc)


if __name__ == "__main__":
    main()
