"""新指標 (ukeyasusa v2 / taiou_capacity v2) の中盤 AUC 検証スクリプト。

## 目的
board_pairs_fixed.npz から中盤スナップショットをサブサンプルし、
新旧 tier1 指標に ukeyasusa / taiou_capacity の差分 (1P - 2P) を加えた場合に
video 単位 holdout (LeaveOneGroupOut) の won-AUC が改善するかを測定する。

## 変更点 (v2)
- 中盤定義: 60-240s (旧 60-120s) → サンプル数 447→3709、動画数 45→88 に拡大。
- サブサンプル上限: MAX_SAMPLE=4000 ペアに制限して実行時間を短縮。
- キャッシュ無効化: 重み・候補生成が変わったため旧キャッシュを使わずに再計算。
- won NaN 除外: 全フィルタ後に NaN を明示除外してクラッシュを防ぐ。

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

# 中盤の定義: 60-240s (video holdout の動画数を確保するため広めに取る)。
# 旧 60-120s では N=447/動画45 しか得られず holdout が不安定だった。
# 60-240s では N=3709/動画88 となり信頼性が向上する。
MIDGAME_SEC_LO: float = 60.0
MIDGAME_SEC_HI: float = 240.0

# サブサンプル上限: 4000 ペアを超える場合はランダムサンプリングして高速化する。
# 全件処理では 46 分かかっていた計算を 10-15 分に短縮する目標。
MAX_SAMPLE: int = 4000
RANDOM_SEED_SAMPLE: int = 42

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
    """board_pairs_fixed.npz から 1P-2P 差分特徴量の DataFrame を構築する。

    高速化: 中盤フィルタ [MIDGAME_SEC_LO, MIDGAME_SEC_HI) + won NaN 除外後の
    インデックスのみを計算する (全 N=20463 ではなく ~3700 件)。
    これにより全件計算 (~24 分) を ~4 分に短縮できる。

    Args:
        data: np.load した NpzFile。
        sim: ChainSimulator インスタンス。

    Returns:
        中盤ペアのみの差分特徴量 DataFrame。
    """
    t_sec: np.ndarray = data["t_sec"]
    won: np.ndarray = data["won"]
    # 中盤マスクを先行計算 (won NaN 除外含む)
    mid_mask = (
        (t_sec >= MIDGAME_SEC_LO)
        & (t_sec < MIDGAME_SEC_HI)
        & ~np.isnan(won.astype(float))
    )
    indices = np.where(mid_mask)[0]
    n_mid = len(indices)
    print(f"[中盤先行フィルタ] 全 {len(t_sec)} → 中盤 {n_mid} 件のみ計算します")

    grids_1p: np.ndarray = data["board_1p"]
    grids_2p: np.ndarray = data["board_2p"]
    video_id: np.ndarray = data["video_id"]
    rows: list[dict[str, Any]] = []
    t0 = time.time()

    for j, i in enumerate(indices):
        if j % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (j + 1) * (n_mid - j) if j > 0 else 0.0
            print(f"  [{j}/{n_mid}] elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)
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
    print(f"[完了] {n_mid} 件 / {time.time() - t0:.1f}s", flush=True)
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


# v2: キャッシュ名を変更して旧キャッシュ (v1 重み) との混在を防ぐ。
CACHE_CSV = PROJ_ROOT / "data" / "indicators_v2" / "study" / "validate_diff_cache_v2.csv"


def _subsample_midgame(mid: pd.DataFrame) -> pd.DataFrame:
    """中盤 DataFrame をサブサンプルする (MAX_SAMPLE 超の場合)。

    動画バランスを維持するために video_id 別に比例サンプリングする。
    MAX_SAMPLE 以下ならそのまま返す。

    Args:
        mid: 中盤フィルタ済み DataFrame。

    Returns:
        サブサンプル後の DataFrame。
    """
    if len(mid) <= MAX_SAMPLE:
        return mid
    frac = MAX_SAMPLE / len(mid)
    sampled = mid.groupby("video_id", group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=RANDOM_SEED_SAMPLE)
        if len(x) > 1 else x
    )
    # 四捨五入誤差で MAX_SAMPLE を少し超える場合があるため上限クリップ
    if len(sampled) > MAX_SAMPLE:
        sampled = sampled.sample(n=MAX_SAMPLE, random_state=RANDOM_SEED_SAMPLE)
    print(f"[サブサンプル] {len(mid)} → {len(sampled)} 行 "
          f"(動画数: {mid['video_id'].nunique()} → {sampled['video_id'].nunique()})")
    return sampled.reset_index(drop=True)


def _load_or_compute(data_path: Path, sim: ChainSimulator) -> pd.DataFrame:
    """キャッシュ CSV があれば読み込み、なければ全ペア計算してキャッシュ保存する。

    v2 キャッシュは旧 v1 キャッシュと別ファイル名で管理する。

    Args:
        data_path: board_pairs_fixed.npz のパス。
        sim: ChainSimulator インスタンス。

    Returns:
        全ペアの差分特徴量 DataFrame。
    """
    if CACHE_CSV.exists():
        print(f"[キャッシュ読み込み] {CACHE_CSV}")
        return pd.read_csv(CACHE_CSV)
    if not data_path.exists():
        print(f"[ERROR] データが見つかりません: {data_path}")
        sys.exit(1)
    print(f"[読み込み] {data_path}")
    data = np.load(data_path, allow_pickle=True)
    print("[指標計算中...] board_pairs_fixed.npz の全ペアを処理します (v2 重み)")
    df = _build_diff_df(data, sim)
    CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_CSV, index=False)
    print(f"[キャッシュ保存] {CACHE_CSV}")
    return df


def main() -> None:
    """検証のエントリポイント。v2: 中盤 60-240s + サブサンプル 4000 ペア。"""
    sim = ChainSimulator()
    df = _load_or_compute(DATA_PATH, sim)

    # 中盤限定 (won NaN を除外済み)
    mid = _filter_midgame(df)
    if len(mid) < 30:
        print("[WARN] 中盤サンプルが 30 未満 → 中止")
        sys.exit(1)

    # サブサンプル (MAX_SAMPLE 超の場合)
    mid = _subsample_midgame(mid)

    report_univariate_auc(mid)
    report_holdout_auc(mid)

    print("\n[完了]")


if __name__ == "__main__":
    main()
