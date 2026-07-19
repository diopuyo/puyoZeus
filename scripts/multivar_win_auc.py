"""多変量 OOF win-AUC 検証スクリプト。

カタログ推奨候補(board sim 5指標 + 関係化指標)を既存指標と組み合わせて
中盤 AUC が向上するか video 単位 GroupKFold OOF で検証する。

条件:
  1. baseline   = 既存指標セット
  2. +board_sim  = baseline + board sim 5 指標
  3. +relational = baseline + 関係化指標
  4. +all        = baseline + board_sim + relational

使い方:
    python -m scripts.multivar_win_auc
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

# =============================================================================
# 定数
# =============================================================================
N_FOLDS: int = 5
MAX_TDIFF_SEC: float = 1.0
BOARD_MATCH_TOL_SEC: float = 2.0
TSUMO_EARLY_RATIO: float = 0.33
TSUMO_LATE_RATIO: float = 0.67

GBC_PARAMS: dict[str, Any] = {
    "max_iter": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "random_state": 42,
    "early_stopping": False,
}
PERM_N_REPEATS: int = 20
PERM_RANDOM_STATE: int = 42

META_COLS: frozenset[str] = frozenset([
    "video_id", "game_idx", "t_sec", "frame", "tsumo", "side",
    "reach_fire_power_source", "chain_duration_source",
    "reach_fire_power_max_chain", "won",
])
REDUNDANT_COLS: frozenset[str] = frozenset([
    "absorption_capacity", "absorption_capacity_raw",
])
BOARD_SIM_INDICATORS: list[str] = [
    "saturated_chain_count",
    "ignition_point_count",
    "multi_color_ignition",
    "sub_chain_count",
    "simultaneous_pop_richness",
]
VIDEO_IDS: list[str] = [f"video_{i}" for i in range(29, 39)]
NPZ_KEYS: list[str] = [f"v{i}" for i in range(29, 39)]


# =============================================================================
# board sim 計算
# =============================================================================

def compute_board_sim(boards_dir: Path) -> "pd.DataFrame | None":
    """boards/v29-v38.npz から board sim 指標を計算する。"""
    try:
        from src.board import Board
        from src.indicators_v2 import (
            ChainSimulator,
            saturated_chain_count,
            ignition_point_count,
            multi_color_ignition,
            sub_chain_count,
            simultaneous_pop_richness,
        )
    except ImportError as exc:
        print(f"[board_sim] src import 失敗: {exc}", flush=True)
        return None

    funcs = [
        ("saturated_chain_count", saturated_chain_count),
        ("ignition_point_count", ignition_point_count),
        ("multi_color_ignition", multi_color_ignition),
        ("sub_chain_count", sub_chain_count),
        ("simultaneous_pop_richness", simultaneous_pop_richness),
    ]
    sim = ChainSimulator()
    rows: list[dict] = []

    for vid, npz_key in zip(VIDEO_IDS, NPZ_KEYS):
        npz_path = boards_dir / f"{npz_key}.npz"
        if not npz_path.exists():
            print(f"[board_sim] {npz_path} 不在 -> スキップ", flush=True)
            continue
        npz = np.load(str(npz_path), allow_pickle=True)
        grids = npz["grids"]
        n = grids.shape[0]
        print(f"[board_sim] {vid}: {n} 盤面計算中...", flush=True)
        for i in range(n):
            board = Board.from_list(grids[i].tolist())
            row: dict = {
                "video_id": vid,
                "side": str(npz["side"][i]),
                "t_sec": float(npz["t_sec"][i]),
                "game_idx": int(npz["game_idx"][i]),
            }
            for fn, func in funcs:
                row[f"{fn}_raw"] = func(board, sim).raw
            rows.append(row)

    if not rows:
        return None
    df = pd.DataFrame(rows)
    print(f"[board_sim] 完了: {len(df)} 行", flush=True)
    return df


def _nearest_match(
    labeled_df: pd.DataFrame,
    board_sim: pd.DataFrame,
) -> pd.DataFrame:
    """labeled_df と board_sim を video_id+side+t_sec 近傍でマッチする。"""
    sim_raw = [f"{fn}_raw" for fn in BOARD_SIM_INDICATORS]
    result = {c: [float("nan")] * len(labeled_df) for c in sim_raw}

    b_grouped = {}
    for (vid, side), grp in board_sim.groupby(["video_id", "side"]):
        b_grouped[(vid, side)] = grp["t_sec"].values, grp[sim_raw].values

    l_arr = labeled_df[["video_id", "side", "t_sec"]].values
    for li, (vid, side, lt) in enumerate(l_arr):
        key = (vid, side)
        if key not in b_grouped:
            continue
        b_ts, b_vals = b_grouped[key]
        diffs = np.abs(b_ts - float(lt))
        best = int(diffs.argmin())
        if diffs[best] <= BOARD_MATCH_TOL_SEC:
            for ci, c in enumerate(sim_raw):
                result[c][li] = float(b_vals[best, ci])

    out = labeled_df.copy()
    for c in sim_raw:
        out[c] = result[c]
    return out



# =============================================================================
# データ準備
# =============================================================================

def load_and_pair(labeled_path: str) -> pd.DataFrame:
    """labeled_win.csv を読み込み 1P/2P ペアリングする。"""
    df = pd.read_csv(labeled_path)
    df = df[df["won"].notna()].copy()
    df["won"] = df["won"].astype(int)
    print(f"[load] won ラベル行: {len(df)}", flush=True)

    p1 = df[df["side"] == "1P"].reset_index(drop=True)
    p2 = df[df["side"] == "2P"].reset_index(drop=True)
    rows: list[dict] = []
    for vid, g1 in p1.groupby("video_id"):
        g2 = p2[p2["video_id"] == vid].reset_index(drop=True)
        if len(g2) == 0:
            continue
        t2 = g2["t_sec"].values
        for _, r1 in g1.iterrows():
            diffs = np.abs(t2 - float(r1["t_sec"]))
            idx_min = int(diffs.argmin())
            if diffs[idx_min] > MAX_TDIFF_SEC:
                continue
            r2 = g2.iloc[idx_min]
            if abs(float(r1["won"]) + float(r2["won"]) - 1.0) > 0.01:
                continue
            merged: dict = {}
            for col in r1.index:
                merged[f"{col}_1p"] = r1[col]
            for col in g2.columns:
                merged[f"{col}_2p"] = r2[col]
            merged["t_diff"] = diffs[idx_min]
            rows.append(merged)
    paired = pd.DataFrame(rows)
    print(f"[pair] ペア成立: {len(paired)} 行", flush=True)
    return paired


def get_baseline_cols(paired: pd.DataFrame) -> list[str]:
    """既存指標のベース列名リストを返す。"""
    all_exclude = META_COLS | REDUNDANT_COLS
    result: list[str] = []
    for col in paired.columns:
        if not col.endswith("_1p"):
            continue
        base = col[:-3]
        if base in all_exclude:
            continue
        if base.endswith("_raw") or base.endswith("_source"):
            continue
        if base == "reach_fire_power_max_chain":
            continue
        if pd.api.types.is_numeric_dtype(paired[col]):
            result.append(base)
    return result


def add_relational_cols(paired: pd.DataFrame) -> list[str]:
    """関係化指標を paired に追加して新規列名リストを返す。"""
    new_cols: list[str] = []
    eps = 1e-6

    def sdiv(a: pd.Series, b: pd.Series) -> pd.Series:
        return a / b.clip(lower=eps)

    paired["death_margin_ratio"] = sdiv(
        paired["death_margin_1p"].astype(float),
        paired["death_margin_2p"].astype(float),
    )
    new_cols.append("death_margin_ratio")

    paired["death_margin_diff"] = (
        paired["death_margin_1p"].astype(float)
        - paired["death_margin_2p"].astype(float)
    )
    new_cols.append("death_margin_diff")

    cr1 = paired.get("current_max_chain_raw_1p",
                     paired.get("current_max_chain_1p", pd.Series(0.0, index=paired.index)))
    cr2 = paired.get("current_max_chain_raw_2p",
                     paired.get("current_max_chain_2p", pd.Series(0.0, index=paired.index)))
    paired["chain_ratio"] = sdiv(cr1.astype(float), cr2.astype(float))
    new_cols.append("chain_ratio")

    cp1 = paired.get("conn_pair_count_1p", pd.Series(float("nan"), index=paired.index))
    cp2 = paired.get("conn_pair_count_2p", pd.Series(float("nan"), index=paired.index))
    paired["conn_pair_diff"] = cp1.astype(float) - cp2.astype(float)
    new_cols.append("conn_pair_diff")

    rfp1 = paired.get("reach_fire_power_1p", pd.Series(float("nan"), index=paired.index))
    abs2 = paired.get("absorption_capacity_raw_2p",
                      paired.get("absorption_capacity_2p",
                                 pd.Series(float("nan"), index=paired.index)))
    paired["reach_capacity_ratio"] = sdiv(rfp1.astype(float), abs2.astype(float))
    new_cols.append("reach_capacity_ratio")

    print(f"[relational] 追加列: {new_cols}", flush=True)
    return new_cols


def add_board_sim_cols(
    paired: pd.DataFrame,
    board_sim: "pd.DataFrame | None",
) -> list[str]:
    """board sim 指標をペアに追加して新規列名リストを返す。"""
    if board_sim is None:
        print("[board_sim_cols] board_sim なし -> スキップ", flush=True)
        return []

    sim_raw = [f"{fn}_raw" for fn in BOARD_SIM_INDICATORS]
    new_cols: list[str] = []

    df_1p = paired[["video_id_1p", "side_1p", "t_sec_1p"]].rename(columns={
        "video_id_1p": "video_id", "side_1p": "side", "t_sec_1p": "t_sec",
    })
    df_2p = paired[["video_id_2p", "side_2p", "t_sec_2p"]].rename(columns={
        "video_id_2p": "video_id", "side_2p": "side", "t_sec_2p": "t_sec",
    })

    matched_1p = _nearest_match(df_1p, board_sim)
    matched_2p = _nearest_match(df_2p, board_sim)

    for c in sim_raw:
        base = c[:-4]
        v1 = matched_1p[c].values.astype(float)
        v2 = matched_2p[c].values.astype(float)
        paired[f"{base}_sim_1p"] = v1
        paired[f"{base}_sim_2p"] = v2
        paired[f"{base}_sim_diff"] = v1 - v2
        new_cols.extend([f"{base}_sim_1p", f"{base}_sim_2p", f"{base}_sim_diff"])

    n_matched = int(paired[new_cols[0]].notna().sum())
    print(f"[board_sim_cols] マッチ行数: {n_matched}/{len(paired)}", flush=True)
    return new_cols



# =============================================================================
# OOF 評価
# =============================================================================

def build_X(
    paired: pd.DataFrame,
    baseline_cols: list[str],
    extra_cols: list[str],
) -> "tuple[np.ndarray, list[str]]": 
    """特徴量行列を構築する。"""
    feat: dict[str, pd.Series] = {}
    for base in baseline_cols:
        for suf in ("_1p", "_2p", "_diff"):
            col = f"{base}{suf}"
            if col in paired.columns:
                feat[col] = paired[col].astype(float)
    for col in extra_cols:
        if col in paired.columns:
            feat[col] = paired[col].astype(float)
    feat_df = pd.DataFrame(feat, index=paired.index)
    return feat_df.fillna(0.0).values, list(feat_df.columns)


def oof_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """GroupKFold OOF AUC を計算する。"""
    n_uni = len(np.unique(groups))
    folds = min(N_FOLDS, max(2, n_uni))
    proba = np.full(len(y), float("nan"))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups=groups):
        m = HistGradientBoostingClassifier(**GBC_PARAMS)
        m.fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]
    valid = ~np.isnan(proba)
    yv, pv = y[valid], proba[valid]
    return float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else float("nan")


def phase_aucs(
    paired: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """位相別 OOF AUC を計算する。"""
    ts = paired["tsumo_1p"].astype(float).values
    q33 = float(np.quantile(ts, TSUMO_EARLY_RATIO))
    q67 = float(np.quantile(ts, TSUMO_LATE_RATIO))
    masks = {
        "序盤": ts <= q33,
        "中盤": (ts > q33) & (ts <= q67),
        "終盤": ts > q67,
    }
    res: dict[str, float] = {}
    for ph, mask in masks.items():
        Xp, yp, gp = X[mask], y[mask], groups[mask]
        if len(Xp) < 20 or len(np.unique(yp)) < 2:
            res[ph] = float("nan")
        else:
            res[ph] = oof_auc(Xp, yp, gp)
    return res


def compute_perm_importance(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feat_names: list[str],
) -> pd.DataFrame:
    """fold 平均 permutation importance を返す。"""
    n_uni = len(np.unique(groups))
    folds = min(N_FOLDS, max(2, n_uni))
    imp_list: list[np.ndarray] = []
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups=groups):
        m = HistGradientBoostingClassifier(**GBC_PARAMS)
        m.fit(X[tr], y[tr])
        perm = permutation_importance(
            m, X[te], y[te],
            n_repeats=PERM_N_REPEATS,
            random_state=PERM_RANDOM_STATE,
            scoring="roc_auc",
        )
        imp_list.append(perm.importances_mean)
    imp = np.array(imp_list)
    return pd.DataFrame({
        "feature": feat_names,
        "importance_mean": imp.mean(axis=0),
        "importance_std": imp.std(axis=0, ddof=1),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)



# =============================================================================
# 出力・保存
# =============================================================================

def _fmt(v: float) -> str:
    return "   n/a  " if (v != v) else f"{v:.4f}"   # nan check


def _delta_str(v: float, base: float) -> str:
    if (v != v) or (base != base):
        return "    n/a "
    return f"{v - base:+.4f}"


def print_results(
    conditions: list[str],
    phase_results: dict[str, dict[str, float]],
    perm_df: pd.DataFrame,
    board_sim_matched: int,
    total_pairs: int,
) -> None:
    """結果を標準出力に整形して出力する。"""
    print()
    print("=" * 80)
    print("  多変量 OOF win-AUC 検証結果")
    print("=" * 80)
    print(f"  ペア行数: {total_pairs}  board_sim マッチ行数: {board_sim_matched}")
    print()

    base_aucs = phase_results.get("baseline", {})
    phases = ["序盤", "中盤", "終盤"]

    print(
        f"  {"条件":<18}  {"序盤AUC":>8}  {"中盤AUC":>8}  {"終盤AUC":>8}"
        f"  {"Δ序盤":>8}  {"Δ中盤":>8}  {"Δ終盤":>8}"
    )
    print("  " + "-" * 74)

    for cond in conditions:
        aucs = phase_results.get(cond, {})
        row_s = f"  {cond:<18}"
        for ph in phases:
            row_s += f"  {_fmt(aucs.get(ph, float("nan"))):>8}"
        if cond == "baseline":
            row_s += "     (base)    (base)    (base)"
        else:
            for ph in phases:
                row_s += f"  {_delta_str(aucs.get(ph, float("nan")), base_aucs.get(ph, float("nan"))):>8}"
        print(row_s)
    print()

    mid_base = base_aucs.get("中盤", float("nan"))
    print(f"  --- 中盤の壁 (baseline 中盤 {_fmt(mid_base)}) 判定 ---")
    for cond in conditions:
        if cond == "baseline":
            continue
        mid = phase_results.get(cond, {}).get("中盤", float("nan"))
        if mid != mid:
            print(f"  {cond}: n/a")
        elif mid > mid_base:
            print(f"  {cond}: [壁越え]  中盤 {mid:.4f}  Δ={mid - mid_base:+.4f}")
        else:
            print(f"  {cond}: [壁越えず] 中盤 {mid:.4f}  Δ={mid - mid_base:+.4f}")
    print()

    print("  --- Permutation Importance Top10 (+all 条件) ---")
    print(f"  {"rank":>4}  {"feature":<50}  {"importance":>11}")
    print("  " + "-" * 70)
    for i, row in perm_df.head(10).iterrows():
        mark = "*" if row["importance_mean"] > 0.001 else " "
        print(f"  {i + 1:>4}{mark} {row["feature"]:<50}  {row["importance_mean"]:>+11.6f}")
    print()


def save_csv(
    conditions: list[str],
    phase_results: dict[str, dict[str, float]],
    perm_df: pd.DataFrame,
    out_path: str,
) -> None:
    """結果を CSV に保存する。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    base_aucs = phase_results.get("baseline", {})
    rows: list[dict] = []
    for cond in conditions:
        aucs = phase_results.get(cond, {})
        row: dict = {"condition": cond}
        for ph in ["序盤", "中盤", "終盤"]:
            cur = aucs.get(ph, float("nan"))
            base = base_aucs.get(ph, float("nan"))
            row[f"auc_{ph}"] = cur
            row[f"delta_{ph}"] = (
                (cur - base) if not ((cur != cur) or (base != base)) else float("nan")
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[save] AUC CSV: {out_path}", flush=True)
    imp_path = str(out).replace(".csv", "_importance.csv")
    perm_df.to_csv(imp_path, index=False)
    print(f"[save] Importance CSV: {imp_path}", flush=True)



# =============================================================================
# メイン
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="多変量 OOF win-AUC 検証")
    parser.add_argument("--labeled", default="data/indicators_v2/study/labeled_win.csv")
    parser.add_argument("--boards-dir", default="data/indicators_v2/boards")
    parser.add_argument("--out", default="data/indicators_v2/multivar_win_auc.csv")
    parser.add_argument("--no-board-sim", action="store_true",
                        help="board sim 計算をスキップ")
    args = parser.parse_args()

    print(f"[main] labeled={args.labeled}", flush=True)
    print(f"[main] boards_dir={args.boards_dir}", flush=True)

    # 1. データ読み込み・ペアリング
    print("\n=== 1. データ読み込み・ペアリング ===", flush=True)
    paired = load_and_pair(args.labeled)
    y = paired["won_1p"].astype(int).values
    groups = paired["video_id_1p"].values
    print(
        f"  won=1: {(y==1).sum()}  won=0: {(y==0).sum()}"
        f"  動画数: {len(np.unique(groups))}",
        flush=True,
    )

    # 2. board sim 指標計算
    print("\n=== 2. board sim 指標計算 ===", flush=True)
    board_sim_df = None
    if not args.no_board_sim:
        board_sim_df = compute_board_sim(Path(args.boards_dir))

    # 3. 関係化指標追加
    print("\n=== 3. 関係化指標追加 ===", flush=True)
    relational_cols = add_relational_cols(paired)

    # 4. board sim 列追加
    print("\n=== 4. board sim 列をペアに追加 ===", flush=True)
    board_sim_cols = add_board_sim_cols(paired, board_sim_df)
    board_sim_matched = (
        int(paired[board_sim_cols[0]].notna().sum()) if board_sim_cols else 0
    )

    # 5. 特徴量セット定義
    baseline_cols = get_baseline_cols(paired)
    print(f"[features] baseline 列数: {len(baseline_cols)}", flush=True)
    conditions = ["baseline", "+relational", "+board_sim", "+all"]
    extra_map: dict[str, list[str]] = {
        "baseline": [],
        "+relational": relational_cols,
        "+board_sim": board_sim_cols,
        "+all": relational_cols + board_sim_cols,
    }

    # 6. 各条件の位相別 OOF AUC
    print("\n=== 5. 各条件 OOF AUC 計算 ===", flush=True)
    phase_results: dict[str, dict[str, float]] = {}
    all_X = None
    all_feat_names: list[str] = []

    for cond in conditions:
        X, feat_names = build_X(paired, baseline_cols, extra_map[cond])
        print(f"\n[{cond}] 特徴量数={len(feat_names)}", flush=True)
        aucs = phase_aucs(paired, X, y, groups)
        phase_results[cond] = aucs
        for ph, v in aucs.items():
            print(f"    {ph}: {_fmt(v)}", flush=True)
        if cond == "+all":
            all_X, all_feat_names = X, feat_names

    # 7. Permutation Importance (+all)
    print("\n=== 6. Permutation Importance (+all) ===", flush=True)
    perm_df = pd.DataFrame()
    if all_X is not None and all_feat_names:
        perm_df = compute_perm_importance(all_X, y, groups, all_feat_names)

    # 8. 出力・保存
    print_results(conditions, phase_results, perm_df, board_sim_matched, len(paired))
    save_csv(conditions, phase_results, perm_df, args.out)


if __name__ == "__main__":
    main()
