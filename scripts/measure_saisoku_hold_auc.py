"""催促保持 (saisoku_hold) v1 の win-AUC 検証。

`reference_saisoku_exchange_model_2026-07-22` に基づき実装した
`src.indicators_v2.saisoku_hold` の3出力 (flag/max_ojama/count) を、
ティア均等23動画 (boards_lean_fixed) の全 STABLE 盤面について計算し、
won との単変量 AUC を ティア別 + 全体 + 位相別 (序盤/中盤/終盤) で測定する。

比較対象: sub_chain_count (副砲・既に+0.014で効いた実績あり) / current_max_chain。
saisoku_hold が won を予測する信号を独自に持つか (特に中盤) を検証し、
sub_chain_count との相関も測って冗長性 (差別化できているか) をチェックする。

対象23動画 (userタスク指定・固定):
    チャレンジャー: c5 c6 c7 c11 c16 c21 c22 c28 c30 c31
    マスター:      c44 c51 c53 c54 c59 c62 c68 c73 c78 c80
    S級:           c82 c83 c84

出力:
    data/indicators_v2/study/saisoku_hold_features.csv (中間: 全行の指標値)
    data/indicators_v2/study/saisoku_hold_auc.csv       (AUC 結果表)
    data/indicators_v2/study/saisoku_hold_corr.csv       (sub_chain_count 相関)

使い方:
    PYTHONPATH=. python -m scripts.measure_saisoku_hold_auc
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# スレッド制限 (熱暴走防止、feedback_thermal_safety_mandatory 準拠)
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "2")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import Board, VALID_COLORS, COLOR_UNKNOWN  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.indicators_v2 import (  # noqa: E402
    saisoku_hold, sub_chain_count, current_max_chain,
)
from scripts.prescreen_candidates import (  # noqa: E402
    univariate_auc, PHASE_ALL, PHASE_EARLY, PHASE_MID, PHASE_LATE,
    TSUMO_EARLY_MAX, TSUMO_LATE_MIN,
)

# ============================
# 定数定義
# ============================

NPZ_DIR: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed"
OUT_DIR: Path = PROJ_ROOT / "data" / "indicators_v2" / "study"
FEATURES_CSV: Path = OUT_DIR / "saisoku_hold_features.csv"
AUC_CSV: Path = OUT_DIR / "saisoku_hold_auc.csv"
CORR_CSV: Path = OUT_DIR / "saisoku_hold_corr.csv"

# 対象23動画のティア対応表 (userタスク指定、measure_exchange_dynamics.py と同一)
TIER_CHALLENGER: tuple[str, ...] = (
    "c5", "c6", "c7", "c11", "c16", "c21", "c22", "c28", "c30", "c31",
)
TIER_MASTER: tuple[str, ...] = (
    "c44", "c51", "c53", "c54", "c59", "c62", "c68", "c73", "c78", "c80",
)
TIER_S_CLASS: tuple[str, ...] = ("c82", "c83", "c84")
TIER_MAP: dict[str, str] = {
    **{v: "チャレンジャー" for v in TIER_CHALLENGER},
    **{v: "マスター" for v in TIER_MASTER},
    **{v: "S級" for v in TIER_S_CLASS},
}
ALL_VIDEOS: tuple[str, ...] = TIER_CHALLENGER + TIER_MASTER + TIER_S_CLASS

# 測定対象の指標列 (raw値でAUCを取る)
FEATURE_COLS: list[str] = [
    "saisoku_hold_flag", "saisoku_hold_max_ojama", "saisoku_hold_count",
    "sub_chain_count", "current_max_chain",
]

# 進捗ログ間隔
LOG_EVERY: int = 2000


def _grid_to_board(grid: np.ndarray) -> Board:
    """shape (13, 6) int8 ndarray を Board オブジェクトに変換する。"""
    board = Board()
    g = grid.astype(np.int64)
    g[~np.isin(g, list(VALID_COLORS) + [9])] = COLOR_UNKNOWN
    board._grid = g.astype(np.uint8)
    return board


def _compute_row_indicators(grid: np.ndarray, sim: ChainSimulator) -> dict[str, float]:
    """1盤面分の指標値 (raw) を計算する。"""
    board = _grid_to_board(grid)
    sh = saisoku_hold(board, sim)
    return {
        "saisoku_hold_flag": sh["saisoku_hold_flag"].raw,
        "saisoku_hold_max_ojama": sh["saisoku_hold_max_ojama"].raw,
        "saisoku_hold_count": sh["saisoku_hold_count"].raw,
        "sub_chain_count": sub_chain_count(board, sim).raw,
        "current_max_chain": current_max_chain(board, sim).raw,
    }


def _load_one_video(video_key: str, sim: ChainSimulator) -> pd.DataFrame:
    """1動画分の npz を読み、全 STABLE 盤面の指標値 + won を計算する。"""
    npz_path = NPZ_DIR / f"{video_key}.npz"
    npz = np.load(npz_path, allow_pickle=True)
    grids = npz["grids"]
    side = npz["side"]
    game_idx = npz["game_idx"]
    t_sec = npz["t_sec"].astype(float)
    won = npz["won"].astype(float)

    n = len(grids)
    rows = [_compute_row_indicators(grids[i], sim) for i in range(n)]
    df = pd.DataFrame(rows)
    df["video_id"] = video_key
    df["tier"] = TIER_MAP[video_key]
    df["side"] = side
    df["game_idx"] = game_idx
    df["t_sec"] = t_sec
    df["won"] = won
    return df


def build_feature_df() -> pd.DataFrame:
    """23動画全件の指標 + won データフレームを構築する。"""
    sim = ChainSimulator()
    frames: list[pd.DataFrame] = []
    t0 = time.time()
    for i, vid in enumerate(ALL_VIDEOS):
        df_v = _load_one_video(vid, sim)
        frames.append(df_v)
        elapsed = time.time() - t0
        print(f"[{i + 1}/{len(ALL_VIDEOS)}] {vid} ({TIER_MAP[vid]}): "
              f"{len(df_v)}行 累計elapsed={elapsed:.1f}s")
    df = pd.concat(frames, ignore_index=True)
    return df


def _attach_phase(df: pd.DataFrame) -> pd.DataFrame:
    """(video_id, side, game_idx) 内の t_sec 進捗率から序盤/中盤/終盤を付与する。"""
    def _progress(g: pd.DataFrame) -> pd.Series:
        tmin, tmax = g["t_sec"].min(), g["t_sec"].max()
        span = max(tmax - tmin, 1e-6)
        return (g["t_sec"] - tmin) / span

    df = df.copy()
    df["progress"] = df.groupby(
        ["video_id", "side", "game_idx"], group_keys=False
    ).apply(_progress)
    conditions = [
        df["progress"] <= TSUMO_EARLY_MAX,
        (df["progress"] > TSUMO_EARLY_MAX) & (df["progress"] <= TSUMO_LATE_MIN),
        df["progress"] > TSUMO_LATE_MIN,
    ]
    df["phase"] = np.select(conditions, ["序盤", "中盤", "終盤"], default="中盤")
    return df


def _phase_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        PHASE_ALL: pd.Series(True, index=df.index),
        PHASE_EARLY: df["phase"] == "序盤",
        PHASE_MID: df["phase"] == "中盤",
        PHASE_LATE: df["phase"] == "終盤",
    }


def compute_auc_table(df: pd.DataFrame) -> pd.DataFrame:
    """ティア別 + 全体 + 位相別の単変量 win-AUC 表を作る (video holdout)。"""
    y = df["won"].astype(float)
    groups = df["video_id"]
    phase_masks = _phase_masks(df)

    tier_subsets: dict[str, pd.Series] = {
        "全体(23本)": pd.Series(True, index=df.index),
        "チャレンジャー": df["tier"] == "チャレンジャー",
        "マスター": df["tier"] == "マスター",
        "S級": df["tier"] == "S級",
    }

    records = []
    for tier_name, tier_mask in tier_subsets.items():
        for phase_name, phase_mask in phase_masks.items():
            mask = tier_mask & phase_mask
            for col in FEATURE_COLS:
                auc = univariate_auc(df[col], y, groups, mask)
                records.append({
                    "tier": tier_name, "phase": phase_name,
                    "feature": col, "auc": auc, "n": int(mask.sum()),
                })
    return pd.DataFrame(records)


def compute_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """saisoku_hold の各出力と sub_chain_count / current_max_chain の相関 (冗長性チェック)。"""
    targets = ["sub_chain_count", "current_max_chain"]
    saisoku_cols = ["saisoku_hold_flag", "saisoku_hold_max_ojama", "saisoku_hold_count"]
    records = []
    for s_col in saisoku_cols:
        for t_col in targets:
            mask = df[s_col].notna() & df[t_col].notna()
            if mask.sum() < 30:
                r = float("nan")
            else:
                r = float(np.corrcoef(df.loc[mask, s_col], df.loc[mask, t_col])[0, 1])
            records.append({"saisoku_col": s_col, "target": t_col,
                            "pearson_r": r, "n": int(mask.sum())})
    return pd.DataFrame(records)


def _print_auc_summary(auc_df: pd.DataFrame) -> None:
    print("\n## win-AUC サマリ (全体・全位相)")
    sub = auc_df[(auc_df["tier"] == "全体(23本)")]
    piv = sub.pivot(index="feature", columns="phase", values="auc")
    cols = [c for c in [PHASE_ALL, PHASE_EARLY, PHASE_MID, PHASE_LATE] if c in piv.columns]
    print(piv[cols].to_string())

    print("\n## win-AUC サマリ (ティア別・中盤)")
    mid = auc_df[auc_df["phase"] == PHASE_MID]
    piv2 = mid.pivot(index="feature", columns="tier", values="auc")
    print(piv2.to_string())


def main() -> None:
    """エントリポイント: 指標計算 -> 位相付与 -> AUC/相関 -> CSV保存。"""
    print("=== measure_saisoku_hold_auc 開始 ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_feature_df()
    df = _attach_phase(df)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"[save] {FEATURES_CSV} ({len(df)}行)")

    print("\n[step] win-AUC 計算中 (video holdout, GroupKFold)...")
    auc_df = compute_auc_table(df)
    auc_df.to_csv(AUC_CSV, index=False)
    print(f"[save] {AUC_CSV}")
    _print_auc_summary(auc_df)

    print("\n[step] sub_chain_count / current_max_chain との相関計算中...")
    corr_df = compute_correlation_table(df)
    corr_df.to_csv(CORR_CSV, index=False)
    print(f"[save] {CORR_CSV}")
    print(corr_df.to_string(index=False))

    print("\n=== measure_saisoku_hold_auc 完了 ===")


if __name__ == "__main__":
    main()
