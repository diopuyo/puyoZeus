"""対応力(taiou_capacity v2)近い地平ラベル検証スクリプト。

exchange_labels.csv の発火イベントに対して、受け手・発火側の盤面を
boards_lean_fixed/*.npz から引き、taiou_capacity / ukeyasusa /
absorption_capacity を計算して taiou_success / opp_buried の
単変量 AUC と増分 AUC (video 単位 holdout) を検証する。

使い方:
    PYTHONPATH=. python -m scripts.validate_taiou_capacity_nearhorizon

出力:
    data/indicators_v2/taiou_capacity_nearhorizon_result.csv
    logs/validate_taiou_nh.log (標準出力をリダイレクト)
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
RESULT_CSV: Path = Path("data/indicators_v2/taiou_capacity_nearhorizon_result.csv")

# 盤面照合の最大時刻差 (秒): これを超えたら欠損扱い
MAX_T_DIFF_SEC: float = 3.0

# video 単位 holdout のフォールド数 (LOO に近いが厳密 LOO は重いため group kfold)
N_FOLDS_VIDEO: int = 5

# スレッド制限 (CPU 節度)
OMP_THREADS: str = "2"
MKL_THREADS: str = "2"

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================
# セクション1: npz インデックスの構築
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
        vids = npz["video_id"]  # shape (N,) str
        # 各 npz 内の video_id は通常 1 種類だが複数になる場合に備え分割
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
                # 複数 npz にまたがる場合は concat
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
    """指定条件に最も近い t_sec の grid を返す。見つからなければ None。

    Args:
        vid_data: _load_npz_index() で得た 1 動画分の辞書。
        side: '1P' または '2P'。
        game_idx: ゲームインデックス (0 始まり)。
        t_sec: 発火イベントの秒数。

    Returns:
        shape (13, 6) int8 ndarray、または None。
    """
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


def _grid_to_board(grid: np.ndarray) -> "Board":  # type: ignore[name-defined]
    """shape (13, 6) int8 ndarray を Board オブジェクトに変換する。"""
    from src.board import Board, VALID_COLORS, COLOR_UNKNOWN
    board = Board()
    # int8 → int に変換し未知色は COLOR_UNKNOWN に正規化
    g = grid.astype(np.int64)
    g[~np.isin(g, list(VALID_COLORS))] = COLOR_UNKNOWN
    board._grid = g.astype(np.uint8)
    return board


def _calc_indicators_one_board(
    grid: np.ndarray | None,
    sim: "ChainSimulator",  # type: ignore[name-defined]
) -> dict[str, float]:
    """1 盤面分の指標辞書を返す。盤面が None なら NaN 埋め。

    計算する指標:
        - taiou_capacity (v2)
        - ukeyasusa
        - absorption_capacity

    Args:
        grid: shape (13, 6) int8 ndarray or None。
        sim: 共有 ChainSimulator インスタンス。

    Returns:
        {'taiou': float, 'ukey': float, 'absorb': float}
    """
    nan = float("nan")
    if grid is None:
        return {"taiou": nan, "ukey": nan, "absorb": nan}
    try:
        from src.indicators_v2 import (
            taiou_capacity,
            ukeyasusa,
            absorption_capacity,
        )
        board = _grid_to_board(grid)
        tc = taiou_capacity(board, simulator=sim)
        uk = ukeyasusa(board, simulator=sim)
        ab = absorption_capacity(board)
        return {
            "taiou": tc.score,
            "ukey": uk.score,
            "absorb": ab.score,
        }
    except Exception as exc:
        logger.debug("指標計算失敗: %s", exc)
        return {"taiou": nan, "ukey": nan, "absorb": nan}


# ============================
# セクション4: 全イベントのバッチ計算
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
    logger.info("イベント数: %d 件の指標計算を開始", total)

    rows_out = []
    t_start = time.time()
    for i, (_, row) in enumerate(labels_df.iterrows()):
        vid = str(row["video_id"])      # 'video_c1'
        game_idx = int(row["game_idx"])
        t_sec = float(row["t_sec"])
        fire_side = str(row["fire_side"])  # '1P' or '2P'
        opp_side_str = _opp_side(fire_side)

        vid_data = npz_index.get(vid)
        if vid_data is None:
            opp_grid = None
            fire_grid = None
        else:
            opp_grid = _find_nearest_grid(vid_data, opp_side_str, game_idx, t_sec)
            fire_grid = _find_nearest_grid(vid_data, fire_side, game_idx, t_sec)

        opp_ind = _calc_indicators_one_board(opp_grid, sim)
        fire_ind = _calc_indicators_one_board(fire_grid, sim)

        rows_out.append({
            **row.to_dict(),
            # 受け手指標 (主役)
            "opp_taiou": opp_ind["taiou"],
            "opp_ukey": opp_ind["ukey"],
            "opp_absorb": opp_ind["absorb"],
            # 発火側指標
            "fire_taiou": fire_ind["taiou"],
            "fire_ukey": fire_ind["ukey"],
            "fire_absorb": fire_ind["absorb"],
            # 差分 (受け手 - 発火側)
            "diff_taiou": opp_ind["taiou"] - fire_ind["taiou"],
            "diff_ukey": opp_ind["ukey"] - fire_ind["ukey"],
            "diff_absorb_new": opp_ind["absorb"] - fire_ind["absorb"],
        })

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
# セクション5: 単変量 AUC 計算
# ============================


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    """NaN を除外して単変量 AUC を返す。計算不能なら 0.5。"""
    from sklearn.metrics import roc_auc_score
    mask = ~np.isnan(score)
    if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
        return float("nan")
    try:
        auc = float(roc_auc_score(y[mask], score[mask]))
        # AUC < 0.5 の場合は反転 (単変量なので絶対値を返す)
        return max(auc, 1.0 - auc)
    except Exception:
        return float("nan")


def compute_univariate_aucs(
    df: pd.DataFrame,
    feature_cols: list[str],
    targets: list[str],
    label: str,
) -> pd.DataFrame:
    """feature_cols x targets の単変量 AUC 表を返す。

    Args:
        df: 特徴量 + ターゲット入り DataFrame。
        feature_cols: AUC を計算する特徴量列名リスト。
        targets: ターゲット列名リスト。
        label: ログ用サブセット名 ('全体' / '中盤' 等)。

    Returns:
        columns=[target, feature, auc, n] の DataFrame。
    """
    records = []
    for target in targets:
        y = df[target].values.astype(float)
        valid_y = ~np.isnan(y)
        for feat in feature_cols:
            score = df[feat].values.astype(float)
            auc = _safe_auc(y[valid_y], score[valid_y])
            n = int(valid_y.sum())
            records.append({"subset": label, "target": target, "feature": feat, "auc": auc, "n": n})
            logger.info("[%s] %s ~ %s AUC=%.4f (n=%d)", label, target, feat, auc, n)
    return pd.DataFrame(records)


# ============================
# セクション6: 増分 AUC (video 単位 holdout)
# ============================


def _video_group_kfold_auc(
    df: pd.DataFrame,
    base_feats: list[str],
    add_feats: list[str],
    target: str,
    n_folds: int,
) -> tuple[float, float]:
    """video 単位 GroupKFold でベース vs ベース+追加の OOF AUC を返す。

    Args:
        df: 特徴量 + ターゲット + 'video_id' 列入り DataFrame。
        base_feats: ベースライン特徴量列名リスト。
        add_feats: 追加する特徴量列名リスト。
        target: ターゲット列名。
        n_folds: GroupKFold の分割数。

    Returns:
        (base_auc, base_plus_add_auc)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    # 有効行 (ターゲット & 全特徴量が NaN でない)
    all_feats = base_feats + add_feats
    valid = df[all_feats + [target, "video_id"]].dropna()
    if len(valid) < 50 or len(valid["video_id"].unique()) < n_folds:
        logger.warning("有効行数不足: %d 行、skipping", len(valid))
        return float("nan"), float("nan")

    X_base = valid[base_feats].values
    X_full = valid[all_feats].values
    y = valid[target].values.astype(float)
    groups = valid["video_id"].values

    gkf = GroupKFold(n_splits=n_folds)
    preds_base = np.full(len(y), float("nan"))
    preds_full = np.full(len(y), float("nan"))

    for tr_idx, va_idx in gkf.split(X_base, y, groups):
        # ベース
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_base[tr_idx])
        X_va = sc.transform(X_base[va_idx])
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X_tr, y[tr_idx])
        preds_base[va_idx] = clf.predict_proba(X_va)[:, 1]
        # ベース + 追加
        sc2 = StandardScaler()
        X_tr2 = sc2.fit_transform(X_full[tr_idx])
        X_va2 = sc2.transform(X_full[va_idx])
        clf2 = LogisticRegression(max_iter=500, C=1.0)
        clf2.fit(X_tr2, y[tr_idx])
        preds_full[va_idx] = clf2.predict_proba(X_va2)[:, 1]

    valid_mask = ~np.isnan(preds_base)
    if valid_mask.sum() < 10 or len(np.unique(y[valid_mask])) < 2:
        return float("nan"), float("nan")
    auc_base = float(roc_auc_score(y[valid_mask], preds_base[valid_mask]))
    auc_full = float(roc_auc_score(y[valid_mask], preds_full[valid_mask]))
    return auc_base, auc_full


def compute_incremental_aucs(
    df: pd.DataFrame,
    targets: list[str],
    label: str,
) -> pd.DataFrame:
    """受けやすさ単体 vs +taiou_capacity の増分 AUC を video holdout で検証。

    Args:
        df: 特徴量 + ターゲット入り DataFrame。
        targets: ターゲット列名リスト。
        label: ログ用サブセット名。

    Returns:
        columns=[subset, target, base_feats, added_feat, base_auc, full_auc, delta_auc, n] の DataFrame。
    """
    # ベース: 受けやすさ系 (受け手の ukey + absorb + opp_absorption_capacity(既存))
    base_feats = ["opp_ukey", "opp_absorb", "opp_absorption_capacity"]
    # 追加: taiou_capacity
    add_feats = ["opp_taiou"]

    records = []
    for target in targets:
        base_auc, full_auc = _video_group_kfold_auc(
            df, base_feats, add_feats, target, N_FOLDS_VIDEO,
        )
        delta = full_auc - base_auc if not (
            np.isnan(full_auc) or np.isnan(base_auc)
        ) else float("nan")
        n = int(df[target].notna().sum())
        records.append({
            "subset": label,
            "target": target,
            "base_feats": "+".join(base_feats),
            "added_feat": "+".join(add_feats),
            "base_auc": base_auc,
            "full_auc": full_auc,
            "delta_auc": delta,
            "n": n,
        })
        logger.info(
            "[%s][%s] 増分AUC: base=%.4f -> full=%.4f (Δ%.4f)",
            label, target, base_auc, full_auc, delta if not np.isnan(delta) else -99,
        )
    return pd.DataFrame(records)


# ============================
# メイン
# ============================


def main() -> None:
    """エントリポイント: 全処理を順次実行し結果 CSV を書き出す。"""
    # CPU スレッド制限 (OMP/MKL)
    os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)
    os.environ.setdefault("MKL_NUM_THREADS", MKL_THREADS)

    logger.info("=== validate_taiou_capacity_nearhorizon 開始 ===")
    logger.info("PYTHONPATH が正しく設定されているか確認...")

    # labels ロード
    labels_df = pd.read_csv(LABELS_CSV)
    logger.info("exchange_labels ロード: %d 行, %d 列", *labels_df.shape)

    # npz インデックス構築
    npz_index = _load_npz_index()

    # 指標計算
    feat_df = build_feature_df(labels_df, npz_index)

    # 結果 CSV 保存 (途中でも利用できるよう先に保存)
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(RESULT_CSV, index=False)
    logger.info("中間結果 CSV 保存: %s", RESULT_CSV)

    # 検証用特徴量
    UNIVAR_FEATS = [
        "opp_taiou", "opp_ukey", "opp_absorb",
        "fire_taiou", "fire_ukey", "fire_absorb",
        "diff_taiou", "diff_ukey", "diff_absorb_new",
        # exchange_labels の既存指標も比較
        "opp_absorption_capacity", "opp_dig_resistance",
        "fire_absorption_capacity", "fire_dig_resistance",
    ]
    TARGETS = ["taiou_success", "opp_buried"]

    auc_rows = []

    # --- 全体 ---
    logger.info("=== 単変量 AUC: 全体 ===")
    auc_rows.append(compute_univariate_aucs(feat_df, UNIVAR_FEATS, TARGETS, "全体"))

    # --- 中盤のみ ---
    mid_df = feat_df[feat_df["phase"] == "中"].copy()
    logger.info("=== 単変量 AUC: 中盤 (n=%d) ===", len(mid_df))
    auc_rows.append(compute_univariate_aucs(mid_df, UNIVAR_FEATS, TARGETS, "中盤"))

    # --- 増分 AUC (全体) ---
    logger.info("=== 増分 AUC: 全体 ===")
    incr_rows = [compute_incremental_aucs(feat_df, TARGETS, "全体")]

    # --- 増分 AUC (中盤) ---
    logger.info("=== 増分 AUC: 中盤 ===")
    incr_rows.append(compute_incremental_aucs(mid_df, TARGETS, "中盤"))

    # 集計 & 保存
    auc_df = pd.concat(auc_rows, ignore_index=True)
    incr_df = pd.concat(incr_rows, ignore_index=True)

    out_auc = RESULT_CSV.with_name("taiou_capacity_nearhorizon_auc.csv")
    out_incr = RESULT_CSV.with_name("taiou_capacity_nearhorizon_incremental.csv")
    auc_df.to_csv(out_auc, index=False)
    incr_df.to_csv(out_incr, index=False)
    logger.info("単変量 AUC 表: %s", out_auc)
    logger.info("増分 AUC 表: %s", out_incr)

    logger.info("=== validate_taiou_capacity_nearhorizon 完了 ===")


if __name__ == "__main__":
    main()
