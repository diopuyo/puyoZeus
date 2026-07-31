"""催促保持 (saisoku_hold) v1 の win-AUC 検証 (参考: labeled_win.csv, v29-38マスターのみ)。

boards_lean_fixed (23動画ティア均等) を主データとする
`scripts.measure_saisoku_hold_auc` の参考として、labeled_win.csv
(v29-38・マスター級10動画・n=40112, won付き15229行) でも同じ指標を測定する。
n が大きい代わりにマスター級のみに偏る点に注意 (ティア均等ではない)。

盤面 grid は data/indicators_v2/boards/v{NN}.npz + _mid + _gap から
_tmp_prescreen_newcand.py の `_load_video_grids` を再利用して取得する。

出力:
    data/indicators_v2/study/saisoku_hold_labeledwin_features.csv
    data/indicators_v2/study/saisoku_hold_labeledwin_auc.csv

使い方:
    PYTHONPATH=. python -m scripts.measure_saisoku_hold_auc_labeledwin
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
from scripts._tmp_prescreen_newcand import _load_video_grids  # noqa: E402

LABELED_CSV: Path = PROJ_ROOT / "data" / "indicators_v2" / "study" / "labeled_win.csv"
OUT_DIR: Path = PROJ_ROOT / "data" / "indicators_v2" / "study"
FEATURES_CSV: Path = OUT_DIR / "saisoku_hold_labeledwin_features.csv"
AUC_CSV: Path = OUT_DIR / "saisoku_hold_labeledwin_auc.csv"

FEATURE_COLS: list[str] = [
    "saisoku_hold_flag", "saisoku_hold_max_ojama", "saisoku_hold_count",
    "sub_chain_count", "current_max_chain",
]


def _grid_to_board(grid: np.ndarray) -> Board:
    board = Board()
    g = grid.astype(np.int64)
    g[~np.isin(g, list(VALID_COLORS) + [9])] = COLOR_UNKNOWN
    board._grid = g.astype(np.uint8)
    return board


def _compute_row_indicators(grid: np.ndarray, sim: ChainSimulator) -> dict[str, float]:
    board = _grid_to_board(grid)
    sh = saisoku_hold(board, sim)
    return {
        "saisoku_hold_flag": sh["saisoku_hold_flag"].raw,
        "saisoku_hold_max_ojama": sh["saisoku_hold_max_ojama"].raw,
        "saisoku_hold_count": sh["saisoku_hold_count"].raw,
        "sub_chain_count": sub_chain_count(board, sim).raw,
        "current_max_chain": current_max_chain(board, sim).raw,
    }


def attach_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """labeled_win.csv 全行に grid 由来の saisoku_hold 系指標を突き合わせて追加する。"""
    sim = ChainSimulator()
    for col in FEATURE_COLS:
        df[col] = np.nan

    videos = sorted(df["video_id"].unique())
    t0 = time.time()
    for vi, vid in enumerate(videos):
        vdata = _load_video_grids(vid)
        if vdata is None:
            print(f"  [WARN] {vid}: npz見つからずスキップ")
            continue
        sub_idx = df.index[df["video_id"] == vid]
        sub = df.loc[sub_idx]
        csv_order = np.lexsort((
            sub["t_sec"].values, sub["game_idx"].values, sub["side"].values))
        sorted_orig_idx = sub_idx.values[csv_order]

        npz_order = np.lexsort((
            vdata["t_sec"], vdata["game_idx"], vdata["side"]))
        grids_sorted = vdata["grids"][npz_order]

        if len(sorted_orig_idx) != len(grids_sorted):
            print(f"  [ERROR] {vid}: 件数不一致 ({len(sorted_orig_idx)} vs "
                  f"{len(grids_sorted)}) スキップ")
            continue

        rows_out = [_compute_row_indicators(grids_sorted[i], sim)
                    for i in range(len(grids_sorted))]
        for col in FEATURE_COLS:
            df.loc[sorted_orig_idx, col] = [r[col] for r in rows_out]

        elapsed = time.time() - t0
        print(f"  [{vi + 1}/{len(videos)}] {vid}: {len(sorted_orig_idx)}行 "
              f"elapsed={elapsed:.1f}s")
    return df


def compute_auc_table(df: pd.DataFrame) -> pd.DataFrame:
    """全体 + 位相別 (tsumo_count_rate 由来) の単変量 win-AUC 表。"""
    df = df[df["won"].notna()].copy()
    y = df["won"].astype(float)
    groups = df["video_id"]
    tcr = df["tsumo_count_rate"].astype(float)
    phase_masks = {
        PHASE_ALL: pd.Series(True, index=df.index),
        PHASE_EARLY: tcr <= TSUMO_EARLY_MAX,
        PHASE_MID: (tcr > TSUMO_EARLY_MAX) & (tcr <= TSUMO_LATE_MIN),
        PHASE_LATE: tcr > TSUMO_LATE_MIN,
    }
    records = []
    for phase_name, mask in phase_masks.items():
        for col in FEATURE_COLS:
            auc = univariate_auc(df[col], y, groups, mask)
            records.append({"phase": phase_name, "feature": col,
                            "auc": auc, "n": int(mask.sum())})
    return pd.DataFrame(records)


def main() -> None:
    """エントリポイント: grid指標付与 -> AUC計算 -> CSV保存。"""
    print("=== measure_saisoku_hold_auc_labeledwin 開始 (参考: v29-38マスターのみ) ===")
    df = pd.read_csv(LABELED_CSV)
    print(f"[load] {LABELED_CSV}: {df.shape}")

    df = attach_indicators(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"[save] {FEATURES_CSV}")

    auc_df = compute_auc_table(df)
    auc_df.to_csv(AUC_CSV, index=False)
    print(f"[save] {AUC_CSV}")
    print("\n## win-AUC (参考: labeled_win.csv v29-38マスターのみ)")
    piv = auc_df.pivot(index="feature", columns="phase", values="auc")
    cols = [c for c in [PHASE_ALL, PHASE_EARLY, PHASE_MID, PHASE_LATE] if c in piv.columns]
    print(piv[cols].to_string())
    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
