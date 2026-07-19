"""新指標 (ukeyasusa / taiou_capacity) の中盤 AUC 検証スクリプト。

## 目的
board_pairs_fixed.npz から中盤スナップショットを抽出し、
新旧 tier1 指標に ukeyasusa / taiou_capacity の差分 (1P - 2P) を加えた場合に
video 単位 holdout (LeaveOneGroupOut) の won-AUC が改善するかを測定する。

## 出力
stdout に:
  (a) 新指標差分単体の単変量 AUC (中盤限定)
  (b) tier1 baseline vs tier1 + 新指標 の video holdout AUC 比較

## 制約
- CPU 節度: スレッド制限 OMP/MKL/OPENBLAS=3
- 重い再収集なし: board_pairs_fixed.npz のみ使用
- 1 関数 50 行以内・型ヒント・日本語コメント・stateless
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

# スレッド制限 (CPU 節度)
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "3")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import Board
from src.chain import ChainSimulator
from src.indicators_v2 import (
    absorption_capacity,
    current_max_chain,
    death_margin,
    dig_resistance,
    immediate_fire_power,
    potential_fire_power,
    taiou_capacity,
    ukeyasusa,
)

# --------------------------------------------------------------------------
# 定数
# --------------------------------------------------------------------------
DATA_PATH = PROJ_ROOT / "data" / "indicators_v2" / "board_pairs_fixed.npz"

# 中盤の定義: t_sec / max(t_sec per game) が 0.34〜0.67 の範囲
# board_pairs_fixed には game_idx があるが t_sec のみで簡易二分割する。
# 上位者対戦の試合時間中央値 ≈ 180 秒。序盤0〜60s / 中盤60〜120s / 終盤120+s。
MIDGAME_SEC_LO: float = 60.0
MIDGAME_SEC_HI: float = 120.0

# GBM パラメータ (軽量)
GBM_PARAMS: dict[str, Any] = {
    "max_iter": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_samples_leaf": 5,
    "random_state": 42,
    "early_stopping": False,
}

# baseline tier1 特徴量名 (1P - 2P 差分として使用)
TIER1_NAMES: list[str] = [
    "absorption_capacity",
    "dig_resistance",
    "death_margin",
    "current_max_chain",
    "potential_fire_power",
    "immediate_fire_power",
]

# 新指標名
NEW_NAMES: list[str] = ["ukeyasusa", "taiou_capacity"]


# --------------------------------------------------------------------------
# 盤面指標計算
# --------------------------------------------------------------------------


def _grid_to_board(grid: np.ndarray) -> Board:
    """numpy (13, 6) int8/int32 盤面を Board オブジェクトに変換する。"""
    return Board.from_list(grid.tolist())


def _compute_indicators(grid: np.ndarray, sim: ChainSimulator) -> dict[str, float]:
    """1 枚の盤面 grid (13×6 int8) から指標辞書を返す (stateless)。"""
    board = _grid_to_board(grid)
    return {
        "absorption_capacity": absorption_capacity(board).score,
        "dig_resistance": dig_resistance(board, sim).score,
        "death_margin": death_margin(board).score,
        "current_max_chain": current_max_chain(board, sim).score,
        "potential_fire_power": potential_fire_power(board, simulator=sim).score,
        "immediate_fire_power": immediate_fire_power(board, simulator=sim).score,
        "ukeyasusa": ukeyasusa(board, sim).score,
        "taiou_capacity": taiou_capacity(board, simulator=sim).score,
    }


def _build_diff_df(data: Any, sim: ChainSimulator) -> pd.DataFrame:
    """board_pairs_fixed.npz から 1P-2P 差分特徴量の DataFrame を構築する。"""
    grids_1p: np.ndarray = data["board_1p"]  # (N, 13, 6)
    grids_2p: np.ndarray = data["board_2p"]  # (N, 13, 6)
    won: np.ndarray = data["won"]            # (N,)
    video_id: np.ndarray = data["video_id"] # (N,)
    t_sec: np.ndarray = data["t_sec"]       # (N,)

    n = len(won)
    rows: list[dict[str, Any]] = []
    t0 = time.time()

    for i in range(n):
        if i % 1000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i)
            print(f"  [{i}/{n}] elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)
        ind_1p = _compute_indicators(grids_1p[i], sim)
        ind_2p = _compute_indicators(grids_2p[i], sim)
        row: dict[str, Any] = {
            "won": float(won[i]),
            "video_id": str(video_id[i]),
            "t_sec": float(t_sec[i]),
        }
        for name in TIER1_NAMES + NEW_NAMES:
            row[f"diff_{name}"] = ind_1p[name] - ind_2p[name]
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[完了] {n} 件 / {time.time() - t0:.1f}s")
    return df


# --------------------------------------------------------------------------
# 中盤フィルタ
# --------------------------------------------------------------------------


def _filter_midgame(df: pd.DataFrame) -> pd.DataFrame:
    """t_sec が [MIDGAME_SEC_LO, MIDGAME_SEC_HI) かつ won が NaN でない行を抽出する。"""
    mask = (
        (df["t_sec"] >= MIDGAME_SEC_LO)
        & (df["t_sec"] < MIDGAME_SEC_HI)
        & df["won"].notna()
    )
    mid = df[mask].copy()
    mid["won"] = mid["won"].astype(float)
    print(f"[中盤フィルタ] {len(df)} → {len(mid)} 行 "
          f"({MIDGAME_SEC_LO:.0f}s〜{MIDGAME_SEC_HI:.0f}s), "
          f"won=1: {int(mid['won'].sum())}, won=0: {int((mid['won']==0).sum())}")
    return mid


# --------------------------------------------------------------------------
# 単変量 AUC
# --------------------------------------------------------------------------


def _univariate_auc(y: np.ndarray, x: np.ndarray) -> float:
    """単変量 AUC (分散ゼロ / 少サンプルは nan)。"""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30 or len(set(y[m].tolist())) < 2 or np.std(x[m]) == 0:
        return float("nan")
    return float(roc_auc_score(y[m], x[m]))


def report_univariate_auc(mid: pd.DataFrame) -> None:
    """(a) 新指標差分の単変量 AUC を出力する。"""
    y = mid["won"].to_numpy(float)
    print("\n=== (a) 単変量 AUC (中盤限定) ===")
    print(f"{'指標差分':<30} {'AUC':>8}")
    for name in TIER1_NAMES + NEW_NAMES:
        col = f"diff_{name}"
        if col not in mid.columns:
            print(f"{col:<30} {'(列なし)':>8}")
            continue
        auc = _univariate_auc(y, mid[col].to_numpy(float))
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "  nan "
        flip_note = " [flip]" if not np.isnan(auc) and auc < 0.5 else ""
        print(f"{col:<30} {auc_str:>8}{flip_note}")


# --------------------------------------------------------------------------
# video 単位 holdout (LeaveOneGroupOut) AUC
# --------------------------------------------------------------------------


def _logo_auc(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
) -> float:
    """LeaveOneGroupOut で OOF 確率を計算し AUC を返す。"""
    oof = np.full(len(y), np.nan)
    logo = LeaveOneGroupOut()
    for tr_idx, te_idx in logo.split(X, y, groups=groups):
        if len(np.unique(y[tr_idx])) < 2:
            continue
        mdl = HistGradientBoostingClassifier(**GBM_PARAMS)
        mdl.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = mdl.predict_proba(X[te_idx])[:, 1]
    valid = np.isfinite(oof)
    if valid.sum() < 30 or len(set(y[valid].tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y[valid], oof[valid]))


def report_holdout_auc(mid: pd.DataFrame) -> None:
    """(b) baseline vs baseline + 新指標の video 単位 holdout AUC を比較出力する。"""
    y = mid["won"].to_numpy(float)
    groups = mid["video_id"].to_numpy()

    baseline_cols = [f"diff_{n}" for n in TIER1_NAMES]
    new_cols = [f"diff_{n}" for n in NEW_NAMES]

    # baseline のみ
    X_base = mid[baseline_cols].fillna(0.0).to_numpy(float)
    auc_base = _logo_auc(X_base, y, groups)

    # baseline + 新指標
    X_plus = mid[baseline_cols + new_cols].fillna(0.0).to_numpy(float)
    auc_plus = _logo_auc(X_plus, y, groups)

    print("\n=== (b) video 単位 holdout AUC (LeaveOneGroupOut, 中盤) ===")
    print(f"  baseline tier1     : {auc_base:.4f}")
    print(f"  tier1 + 新指標 2本 : {auc_plus:.4f}")
    delta = auc_plus - auc_base if not (np.isnan(auc_base) or np.isnan(auc_plus)) else float("nan")
    delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "  nan "
    print(f"  ΔAUC               : {delta_str}")
    print(f"  動画数: {mid['video_id'].nunique()}")


# --------------------------------------------------------------------------
# メイン
# --------------------------------------------------------------------------


CACHE_CSV = PROJ_ROOT / "data" / "indicators_v2" / "study" / "validate_diff_cache.csv"


def main() -> None:
    """検証のエントリポイント。キャッシュ CSV があれば再利用する。"""
    if CACHE_CSV.exists():
        print(f"[キャッシュ読み込み] {CACHE_CSV}")
        df = pd.read_csv(CACHE_CSV)
    else:
        if not DATA_PATH.exists():
            print(f"[ERROR] データが見つかりません: {DATA_PATH}")
            sys.exit(1)
        print(f"[読み込み] {DATA_PATH}")
        data = np.load(DATA_PATH, allow_pickle=True)
        sim = ChainSimulator()
        print("[指標計算中...] board_pairs_fixed.npz の全ペアを処理します")
        df = _build_diff_df(data, sim)
        # CSV キャッシュ保存
        CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_CSV, index=False)
        print(f"[キャッシュ保存] {CACHE_CSV}")

    # 中盤限定 (won NaN を除外済み)
    mid = _filter_midgame(df)
    if len(mid) < 30:
        print("[WARN] 中盤サンプルが 30 未満 → 中止")
        sys.exit(1)

    report_univariate_auc(mid)
    report_holdout_auc(mid)

    print("\n[完了]")


if __name__ == "__main__":
    main()
