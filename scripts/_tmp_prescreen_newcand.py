"""安価な新規指標候補 (holes/herfindahl/shannon/flow/vspace) の実データ AUC 先行スクリーニング。

対象候補 (INDICATOR_CANDIDATES_2026-07-20.md 由来、実装ゼロ/激安枠):
    M26 埋没穴数 (holes)          - 盤面 grid から直接計算
    M1  連結集中度 (Herfindahl)   - 盤面 grid から直接計算 (既存 conn_* とは別軸)
    M9  色均等度 (Shannon evenness) - 盤面 grid から直接計算
    M22 流動性 (immediate_fire_power / saturated_chain_count) - 既存 CSV 列の除算のみ
    M30 縦空間支配 (13 - max_column_height, 対戦相手比)         - 既存 CSV 列の関係化のみ
    N3  配置エントロピー簡易版 (列高さ分散の逆数)                - 盤面 grid から直接計算 (ボーナス)

盤面 grid は data/indicators_v2/boards/v{NN}.npz + _mid + _gap の 3 分割が
labeled_win.csv の全行と厳密に 1対1 対応する (2026-07-21 全被覆再収集済、
video 単位で件数完全一致を確認済)。(side, game_idx, t_sec) の辞書式ソートで
CSV 側・npz 側を同じ順序に揃えて突き合わせる。

video 単位 holdout 単変量 win-AUC を位相別 (序盤/中盤/終盤) に測定し、
既存指標との Pearson 相関 (死票判定材料) も出力する。

出力:
    data/indicators_v2/study/labeled_win_newcand.csv  (新候補列付き中間データ)
    data/indicators_v2/prescreen_newcand_auc.csv      (AUC 結果表)
    data/indicators_v2/prescreen_newcand_corr.csv     (既存指標との相関表)

使い方:
    PYTHONPATH=. python -m scripts._tmp_prescreen_newcand
"""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prescreen_candidates import (
    load_and_pair, univariate_auc, eval_candidate,
    TSUMO_EARLY_MAX, TSUMO_LATE_MIN,
    PHASE_ALL, PHASE_EARLY, PHASE_MID, PHASE_LATE,
)

LABELED_CSV = Path("data/indicators_v2/study/labeled_win.csv")
BOARDS_DIR = Path("data/indicators_v2/boards")
OUT_MERGED_CSV = Path("data/indicators_v2/study/labeled_win_newcand.csv")
OUT_AUC_CSV = Path("data/indicators_v2/prescreen_newcand_auc.csv")
OUT_CORR_CSV = Path("data/indicators_v2/prescreen_newcand_corr.csv")

BOARD_ROWS = 13
BOARD_COLS = 6
COLOR_EMPTY = 0
VALID_COLORS = [1, 2, 3, 4, 5]
EPS = 1.0

NEWCAND_RAW_COLS = [
    "holes_raw", "herf_raw", "shannon_raw", "colheight_invvar_raw",
]

CORR_TARGETS = {
    "holes_raw": ["board_puyo_total_raw", "death_margin_raw", "max_column_height_raw"],
    "herf_raw": ["main_linked_ratio_raw", "conn_max_group_size", "conn_pair_count"],
    "shannon_raw": ["board_color_puyo_total_raw", "conn_pair_count"],
    "flow_raw": ["current_max_chain_raw", "immediate_fire_power_raw", "reach_fire_power_raw"],
    "colheight_invvar_raw": ["column_bumpiness_raw", "max_column_height_raw"],
}


def compute_grid_indicators(grid: np.ndarray) -> dict[str, float]:
    """1盤面分の holes / herf / shannon / colheight_invvar を計算する。

    Args:
        grid: shape (13, 6) int8 ndarray。

    Returns:
        holes_raw / herf_raw / shannon_raw / colheight_invvar_raw を持つ辞書。
    """
    g = grid.astype(np.int64)

    holes = 0
    col_heights: list[int] = []
    for c in range(BOARD_COLS):
        col = g[:, c]
        filled_rows = np.where(col != COLOR_EMPTY)[0]
        if filled_rows.size == 0:
            col_heights.append(0)
            continue
        top_row = int(filled_rows.min())
        col_heights.append(BOARD_ROWS - top_row)
        below = col[top_row + 1:]
        holes += int(np.sum(below == COLOR_EMPTY))

    var = float(np.var(np.array(col_heights, dtype=np.float64)))
    colheight_invvar = 1.0 / (1.0 + var)

    visited = np.zeros_like(g, dtype=bool)
    comp_sizes: list[int] = []
    color_counts = np.zeros(len(VALID_COLORS), dtype=np.int64)
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if visited[r, c]:
                continue
            color = int(g[r, c])
            if color not in VALID_COLORS:
                visited[r, c] = True
                continue
            stack = [(r, c)]
            visited[r, c] = True
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS and not visited[nr, nc]:
                        if int(g[nr, nc]) == color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
            comp_sizes.append(size)
            color_counts[VALID_COLORS.index(color)] += size

    total_puyo = int(color_counts.sum())
    if total_puyo == 0:
        herf = float("nan")
        shannon = float("nan")
    else:
        sizes_arr = np.array(comp_sizes, dtype=np.float64)
        herf = float(np.sum(sizes_arr ** 2) / (total_puyo ** 2))
        probs = color_counts[color_counts > 0].astype(np.float64) / total_puyo
        entropy = float(-np.sum(probs * np.log(probs)))
        shannon = entropy / np.log(len(VALID_COLORS))

    return {
        "holes_raw": float(holes),
        "herf_raw": herf,
        "shannon_raw": shannon,
        "colheight_invvar_raw": colheight_invvar,
    }


def _load_video_grids(video_id: str) -> dict[str, np.ndarray] | None:
    """video_id (例: video_29) に対応する全 npz 分割 (base/_mid/_gap) を連結する。"""
    short = video_id.replace("video_", "v")
    files = sorted(glob.glob(str(BOARDS_DIR / (short + ".npz")))) + \
        sorted(glob.glob(str(BOARDS_DIR / (short + "_*.npz"))))
    if not files:
        return None
    grids, sides, gidx, tsec = [], [], [], []
    for f in files:
        npz = np.load(f, allow_pickle=True)
        grids.append(npz["grids"])
        sides.append(npz["side"])
        gidx.append(npz["game_idx"])
        tsec.append(npz["t_sec"])
    return {
        "grids": np.concatenate(grids),
        "side": np.concatenate(sides),
        "game_idx": np.concatenate(gidx),
        "t_sec": np.concatenate(tsec),
    }


def attach_grid_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """labeled_win.csv 全行に対して grid 由来の新候補列を突き合わせて追加する。"""
    for col in NEWCAND_RAW_COLS:
        df[col] = np.nan

    videos = sorted(df["video_id"].unique())
    t0 = time.time()
    max_t_mismatch = 0.0
    for vi, vid in enumerate(videos):
        vdata = _load_video_grids(vid)
        if vdata is None:
            print("  [WARN] " + vid + ": npz見つからずスキップ")
            continue
        sub_idx = df.index[df["video_id"] == vid]
        sub = df.loc[sub_idx]
        csv_order = np.lexsort((
            sub["t_sec"].values, sub["game_idx"].values, sub["side"].values))
        sorted_orig_idx = sub_idx.values[csv_order]

        npz_order = np.lexsort((
            vdata["t_sec"], vdata["game_idx"], vdata["side"]))
        grids_sorted = vdata["grids"][npz_order]
        t_sorted = vdata["t_sec"][npz_order]

        if len(sorted_orig_idx) != len(grids_sorted):
            print("  [ERROR] " + vid + ": 件数不一致 スキップ")
            continue

        t_csv_sorted = df.loc[sorted_orig_idx, "t_sec"].values
        tdiff = float(np.max(np.abs(t_csv_sorted - t_sorted))) if len(t_sorted) else 0.0
        max_t_mismatch = max(max_t_mismatch, tdiff)

        rows_out = [compute_grid_indicators(grids_sorted[i]) for i in range(len(grids_sorted))]
        for col in NEWCAND_RAW_COLS:
            df.loc[sorted_orig_idx, col] = [r[col] for r in rows_out]

        elapsed = time.time() - t0
        print("  [%d/%d] %s: %d 行 t_diff_max=%.4f elapsed=%.1fs" % (
            vi + 1, len(videos), vid, len(sorted_orig_idx), tdiff, elapsed))

    print("[attach] 全体 t_diff_max=%.4f (許容: <0.01 が理想)" % max_t_mismatch)
    return df


def build_newcand_relational(paired: pd.DataFrame) -> pd.DataFrame:
    """新候補の差分/比の関係的特徴量を構築する。"""
    feats: dict[str, pd.Series] = {}

    def s(col: str, side: str) -> pd.Series:
        key = col + "_" + side
        if key in paired.columns:
            return paired[key].astype(float)
        return pd.Series(np.nan, index=paired.index)

    feats["holes_diff"] = s("holes_raw", "1p") - s("holes_raw", "2p")
    feats["holes_ratio"] = s("holes_raw", "1p") / s("holes_raw", "2p").clip(lower=EPS)

    feats["herf_diff"] = s("herf_raw", "1p") - s("herf_raw", "2p")
    feats["herf_ratio"] = s("herf_raw", "1p") / s("herf_raw", "2p").clip(lower=EPS / 100)

    feats["shannon_diff"] = s("shannon_raw", "1p") - s("shannon_raw", "2p")
    feats["shannon_ratio"] = s("shannon_raw", "1p") / s("shannon_raw", "2p").clip(lower=EPS / 100)

    flow_1p = s("immediate_fire_power_raw", "1p") / s("saturated_chain_count_raw", "1p").clip(lower=EPS)
    flow_2p = s("immediate_fire_power_raw", "2p") / s("saturated_chain_count_raw", "2p").clip(lower=EPS)
    feats["flow_raw_1p"] = flow_1p
    feats["flow_raw_2p"] = flow_2p
    feats["flow_diff"] = flow_1p - flow_2p
    feats["flow_ratio"] = flow_1p / flow_2p.clip(lower=EPS / 100)

    vsp_1p = 13.0 - s("max_column_height_raw", "1p")
    vsp_2p = 13.0 - s("max_column_height_raw", "2p")
    feats["vspace_raw_1p"] = vsp_1p
    feats["vspace_raw_2p"] = vsp_2p
    feats["vspace_diff"] = vsp_1p - vsp_2p
    feats["vspace_ratio"] = vsp_1p / vsp_2p.clip(lower=EPS / 100)

    feats["colheight_invvar_diff"] = s("colheight_invvar_raw", "1p") - s("colheight_invvar_raw", "2p")
    feats["colheight_invvar_ratio"] = (
        s("colheight_invvar_raw", "1p") / s("colheight_invvar_raw", "2p").clip(lower=EPS / 100))

    return pd.DataFrame(feats, index=paired.index)


RAW_SIDE_CANDIDATES = [
    "holes_raw", "herf_raw", "shannon_raw", "colheight_invvar_raw",
]


def main() -> None:
    """エントリポイント: grid指標計算 -> 相関チェック -> ペアリング -> AUC測定。"""
    print("=== _tmp_prescreen_newcand 開始 ===")
    df = pd.read_csv(LABELED_CSV)
    print("[load] %s: %s" % (LABELED_CSV, df.shape,))

    print("[step1] 盤面gridから新候補raw列を計算 (holes/herf/shannon/colheight_invvar)...")
    df = attach_grid_indicators(df)

    OUT_MERGED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_MERGED_CSV, index=False)
    print("[save] " + str(OUT_MERGED_CSV))

    print("[step2] 既存指標とのPearson相関を計算...")
    corr_rows = []
    for cand, targets in CORR_TARGETS.items():
        if cand == "flow_raw":
            src = (df["immediate_fire_power_raw"]
                   / df["saturated_chain_count_raw"].clip(lower=EPS))
        else:
            src = df[cand]
        for tgt in targets:
            if tgt not in df.columns:
                continue
            mask = src.notna() & df[tgt].notna()
            if mask.sum() < 30:
                corr_val = float("nan")
            else:
                corr_val = float(np.corrcoef(src[mask], df[tgt][mask])[0, 1])
            corr_rows.append({"candidate": cand, "existing_col": tgt,
                              "pearson_r": corr_val, "n": int(mask.sum())})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_CORR_CSV, index=False)
    print("[save] " + str(OUT_CORR_CSV))
    print(corr_df.to_string(index=False))

    print("[step3] 1P/2Pペアリング...")
    paired = load_and_pair(str(OUT_MERGED_CSV))

    won_cols = [c for c in paired.columns if c.startswith("won") and c.endswith("_1p")]
    y = paired[won_cols[0]].astype(float)
    groups = paired["video_id_1p"]
    tcr = paired["tsumo_count_rate_1p"].astype(float)
    phase_masks = {
        PHASE_ALL: pd.Series(True, index=paired.index),
        PHASE_EARLY: tcr <= TSUMO_EARLY_MAX,
        PHASE_MID: (tcr > TSUMO_EARLY_MAX) & (tcr <= TSUMO_LATE_MIN),
        PHASE_LATE: tcr > TSUMO_LATE_MIN,
    }
    for ph, m in phase_masks.items():
        print("  位相 %s: %d 行" % (ph, int(m.sum())))

    print("[step4] 生値(自サイド)AUC...")
    auc_rows = []
    for col in RAW_SIDE_CANDIDATES:
        key = col + "_1p"
        if key not in paired.columns:
            continue
        aucs = eval_candidate(paired[key].astype(float), y, groups, phase_masks)
        auc_rows.append(dict({"candidate": col + "(自側生値)", "kind": "raw_self"}, **aucs))

    print("[step5] 関係的特徴量(差/比)AUC...")
    rel = build_newcand_relational(paired)
    for col in rel.columns:
        aucs = eval_candidate(rel[col], y, groups, phase_masks)
        kind = "ratio" if col.endswith("_ratio") else "diff"
        auc_rows.append(dict({"candidate": col, "kind": kind}, **aucs))

    result_df = pd.DataFrame(auc_rows)
    result_df = result_df.sort_values(PHASE_MID, ascending=False, na_position="last")
    OUT_AUC_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_AUC_CSV, index=False)
    print("[save] " + str(OUT_AUC_CSV))

    print("")
    print("## 全候補AUC (中盤ソート)")
    print(result_df.to_string(index=False))
    print("")
    print("=== 完了 ===")


if __name__ == "__main__":
    main()
