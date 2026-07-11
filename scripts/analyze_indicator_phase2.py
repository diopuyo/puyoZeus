"""指標 v2 -- 盤面ぷよ数位相 + 偏相関分析。

## 追加1: 盤面ぷよ数ベースの位相軸
ペアリング後の max(1P.board_color_puyo_total_raw, 2P.board_color_puyo_total_raw) で位相を決定:
- opening : max(手数1P, 手数2P) <= OPENING_TSUMO_MAX  (最優先・別枠)
- 序盤    : max_puyo <= EARLY_PUYO_MAX  (opening でない場合)
- 中盤    : EARLY_PUYO_MAX < max_puyo <= LATE_PUYO_MIN
- 終盤    : max_puyo > LATE_PUYO_MIN

proxy と future_gain(K=3s) の両方について盤面ぷよ数位相別 Spearman 相関を集計。
既存の手数三分位 位相と並べて出力する。

## 追加2: 偏相関 (盤面ぷよ数を制御)
火力系指標 + 対照系指標について:
- board_color_puyo_total_raw を制御変数として偏 Spearman 相関を計算
- 生相関と並べて「独立寄与」が残るか判定

## 使い方
    python -m scripts.analyze_indicator_phase2 --study data/indicators_v2/study
    python -m scripts.analyze_indicator_phase2 --study data/indicators_v2/study --out result.csv

## 注意事項
- _mid.csv は除外 (重複データを避けるため)
- corr_20v.csv は除外 (中間集計ファイル)
- 偏 Spearman = rank 変換後の偏相関 (線形代数実装、安価で頑健)
- 終盤 (>55) は少数サンプル → 解釈は慎重に
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

# ========================================================================
# 定数 (proxy 重みは analyze_indicator_proxy.py と統一)
# ========================================================================

PROXY_W_OJAMA: float = 0.7
PROXY_W_DEATH: float = 0.3

DEFAULT_MAX_TDIFF: float = 1.0   # ペアリング最大時刻差 (秒)
FDR_ALPHA: float = 0.05          # Benjamini-Hochberg FDR 有意水準

# --- 盤面ぷよ数位相境界 ---
OPENING_TSUMO_MAX: int = 12        # max(手数1P, 手数2P) <= この値 → opening
EARLY_PUYO_MAX: int = 20           # max_puyo <= 20 → 序盤
LATE_PUYO_MIN: int = 55            # max_puyo > 55  → 終盤
# 中盤: EARLY_PUYO_MAX < max_puyo <= LATE_PUYO_MIN

# --- forward 分析パラメータ ---
LOOKAHEAD_K_SEC: int = 3           # proxy との対比用 K=3s を主として使う
LOOKAHEAD_TOLERANCE: float = 1.5   # K秒後スナップショット許容誤差 (秒)

# --- 偏相関の制御変数 ---
PARTIAL_CONTROL_COL: str = "board_color_puyo_total_raw"

# --- 火力系指標 (analyze_indicator_forward.py と同一) ---
FIRE_POWER_COLS: list[str] = [
    "current_max_chain",
    "immediate_fire_power",
    "reach_fire_power",
    "chain_efficiency",
    "second_chain_potential",
    "min_puyos_to_ignite",
    "conn_pair_count",
    "conn_triple_count",
    "conn_max_group_size",
]

# --- 対照系指標 ---
CONTROL_COLS: list[str] = [
    "board_ojama_count",
    "max_column_height",
]

# --- proxy 除外列 ---
PROXY_COMPONENTS: frozenset[str] = frozenset([
    "ojama_net_balance", "ojama_net_balance_raw",
    "ojama_forecast", "ojama_forecast_raw",
    "death_margin", "death_margin_raw",
    "death_margin_neighbor", "death_margin_neighbor_raw",
])


# ========================================================================
# データ読み込み
# ========================================================================

def load_study_csvs(study_dir: str) -> pd.DataFrame:
    """study ディレクトリの全 CSV を結合して返す (_mid.csv / corr_20v.csv を除外)。"""
    paths = sorted(Path(study_dir).glob("*.csv"))
    paths = [
        p for p in paths
        if p.name != "corr_20v.csv" and not p.name.endswith("_mid.csv")
    ]
    if not paths:
        raise FileNotFoundError(f"CSV が見つかりません: {study_dir}")
    dfs: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        dfs.append(df)
        print(f"  読み込み: {p.name}  {df.shape[0]} 行")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  合計: {combined.shape[0]} 行, {combined.shape[1]} 列, 動画数: {combined['video_id'].nunique()}")
    return combined


# ========================================================================
# ペアリング
# ========================================================================

def pair_sides(df: pd.DataFrame, max_tdiff: float) -> tuple[pd.DataFrame, dict]:
    """1P / 2P を (video_id, game_idx) 内で時刻最近傍マッチ。"""
    p1 = df[df["side"] == "1P"].reset_index(drop=True)
    p2 = df[df["side"] == "2P"].reset_index(drop=True)
    rows: list[dict] = []
    for (vid, gidx), g1 in p1.groupby(["video_id", "game_idx"]):
        g2 = p2[(p2["video_id"] == vid) & (p2["game_idx"] == gidx)].reset_index(drop=True)
        if len(g2) == 0:
            continue
        t2 = g2["t_sec"].values
        for _, r1 in g1.iterrows():
            diffs = np.abs(t2 - r1["t_sec"])
            idx_min = int(diffs.argmin())
            if diffs[idx_min] <= max_tdiff:
                merged_row: dict = {}
                for col in r1.index:
                    merged_row[f"{col}_1p"] = r1[col]
                for col in g2.columns:
                    merged_row[f"{col}_2p"] = g2.iloc[idx_min][col]
                merged_row["t_diff"] = diffs[idx_min]
                rows.append(merged_row)
    paired = pd.DataFrame(rows)
    total_1p = len(p1)
    pair_rate = len(paired) / total_1p if total_1p > 0 else 0.0
    stats_info: dict = {
        "total_1p_rows": total_1p,
        "paired_rows": len(paired),
        "pair_rate": pair_rate,
        "t_diff_median": paired["t_diff"].median() if len(paired) > 0 else float("nan"),
        "max_tdiff_threshold": max_tdiff,
    }
    return paired, stats_info


# ========================================================================
# proxy 計算
# ========================================================================

def _zscore(s: pd.Series) -> pd.Series:
    """標準化 (std==0 は全 0 で返す)。"""
    std = float(s.std(ddof=1))
    if std < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def compute_proxy(paired: pd.DataFrame) -> pd.Series:
    """優勢 proxy = 0.7*z(ojama_net) + 0.3*z(dm_diff)。"""
    ojama_raw = paired["ojama_net_balance_raw_1p"].astype(float)
    dm_diff = (
        paired["death_margin_raw_1p"].astype(float)
        - paired["death_margin_raw_2p"].astype(float)
    )
    return PROXY_W_OJAMA * _zscore(ojama_raw) + PROXY_W_DEATH * _zscore(dm_diff)


# ========================================================================
# 盤面ぷよ数位相の割り当て
# ========================================================================

def assign_puyo_phase(paired: pd.DataFrame) -> pd.Series:
    """
    各ペア行に盤面ぷよ数位相ラベルを付ける。

    優先順位:
      1. max(tsumo_1p, tsumo_2p) <= OPENING_TSUMO_MAX → "opening"
      2. max(puyo_1p, puyo_2p) <= EARLY_PUYO_MAX       → "序盤"
      3. max(puyo_1p, puyo_2p) <= LATE_PUYO_MIN        → "中盤"
      4. それ以外                                        → "終盤"

    「opening」は手数が両者ともまだ序盤 (max <= 12) の状態を指す。
    連鎖直後にぷよが減っても手数はリセットしないため、
    手数条件を優先することで「本当の試合開始」を識別できる。
    """
    tsumo_max = np.maximum(
        paired["tsumo_1p"].astype(float).values,
        paired["tsumo_2p"].astype(float).values,
    )
    puyo_1p = paired["board_color_puyo_total_raw_1p"].astype(float).values
    puyo_2p = paired["board_color_puyo_total_raw_2p"].astype(float).values
    puyo_max = np.maximum(puyo_1p, puyo_2p)

    # デフォルト: 中盤
    labels = np.full(len(paired), "中盤", dtype=object)
    # 終盤: ぷよ数が上限超え
    labels[puyo_max > LATE_PUYO_MIN] = "終盤"
    # 序盤: ぷよ数が下限以下
    labels[puyo_max <= EARLY_PUYO_MAX] = "序盤"
    # opening: 手数最優先で上書き (連鎖後ぷよ減少状態でも手数が残っていれば opening)
    labels[tsumo_max <= OPENING_TSUMO_MAX] = "opening"

    return pd.Series(labels, index=paired.index, dtype=str)


# ========================================================================
# 手数三分位位相の割り当て (既存スクリプトと同一ロジック)
# ========================================================================

def assign_tsumo_phase(paired: pd.DataFrame) -> pd.Series:
    """手数三分位による位相ラベル (序盤/中盤/終盤)。"""
    tsumo = paired["tsumo_1p"].astype(float)
    q33 = float(tsumo.quantile(0.33))
    q67 = float(tsumo.quantile(0.67))
    labels = np.where(
        tsumo <= q33, "序盤",
        np.where(tsumo <= q67, "中盤", "終盤"),
    )
    return pd.Series(labels, index=paired.index, dtype=str)


# ========================================================================
# Spearman 相関ユーティリティ
# ========================================================================

def spearman_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Spearman (r, p, n_valid)。サンプル不足は (nan, nan, 0)。"""
    mask = ~(np.isnan(x) | np.isnan(y))
    n = int(mask.sum())
    if n < 5:
        return float("nan"), float("nan"), n
    r, p = stats.spearmanr(x[mask], y[mask], nan_policy="omit")
    return float(r), float(p), n


# ========================================================================
# 偏 Spearman 相関 (rank 変換後に残差法)
# ========================================================================

def partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[float, float, int]:
    """
    z を制御した x と y の偏 Spearman 相関を計算する。

    手順:
      1. x, y, z を rank 変換 (ties='average')
      2. rank_x ~ rank_z, rank_y ~ rank_z で OLS 残差を得る
      3. 残差同士の Pearson r が偏 Spearman に相当 (近似)
      4. p 値は t 分布から (自由度 n-3)

    注意: 完全な semi-partial ではなく偏相関 (partial corr)。
    z が x/y に線形の影響を持つ前提の近似。
    """
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    n = int(mask.sum())
    if n < 8:
        # 制御変数込みで自由度 n-3 >= 5 を確保する最低ライン
        return float("nan"), float("nan"), n

    rx = stats.rankdata(x[mask]).astype(float)
    ry = stats.rankdata(y[mask]).astype(float)
    rz = stats.rankdata(z[mask]).astype(float)

    def ols_residuals(dep: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """OLS の残差ベクトルを返す (切片あり)。"""
        A = np.column_stack([np.ones(len(pred)), pred])
        coef, _, _, _ = np.linalg.lstsq(A, dep, rcond=None)
        return dep - A @ coef

    res_x = ols_residuals(rx, rz)
    res_y = ols_residuals(ry, rz)

    if res_x.std() < 1e-9 or res_y.std() < 1e-9:
        return 0.0, 1.0, n

    r, _ = stats.pearsonr(res_x, res_y)
    r = float(r)

    # t 統計量 (自由度 n - 2 - 1 = n - 3)
    df_resid = n - 3
    if df_resid < 1:
        return r, float("nan"), n
    t_stat = r * np.sqrt(df_resid / (1.0 - r ** 2 + 1e-15))
    p_val = float(2 * stats.t.sf(np.abs(t_stat), df=df_resid))
    return r, p_val, n


# ========================================================================
# 位相別相関計算
# ========================================================================

def compute_phase_correlations_by_puyo(
    paired: pd.DataFrame,
    y: np.ndarray,
    phase_labels: pd.Series,
    target_cols: list[str],
    side_suffix: str = "_1p",
) -> dict[str, dict[str, float]]:
    """盤面ぷよ数位相別に Spearman r を計算する。"""
    phase_order = ["opening", "序盤", "中盤", "終盤"]
    result: dict[str, dict[str, float]] = {p: {} for p in phase_order}
    for phase in phase_order:
        mask = (phase_labels == phase).values
        y_phase = y[mask]
        for col in target_cols:
            col_s = f"{col}{side_suffix}"
            if col_s not in paired.columns:
                continue
            x_phase = paired[col_s].astype(float).values[mask]
            r, _, _ = spearman_safe(x_phase, y_phase)
            result[phase][col] = r
    return result


def count_phase_samples(phase_labels: pd.Series) -> dict[str, int]:
    """各位相のサンプル数を返す。"""
    counts = phase_labels.value_counts().to_dict()
    return {p: counts.get(p, 0) for p in ["opening", "序盤", "中盤", "終盤"]}


def compute_phase_correlations_by_tsumo(
    paired: pd.DataFrame,
    y: np.ndarray,
    tsumo_phase_labels: pd.Series,
    target_cols: list[str],
    side_suffix: str = "_1p",
) -> dict[str, dict[str, float]]:
    """手数三分位位相別 Spearman r。"""
    result: dict[str, dict[str, float]] = {"序盤": {}, "中盤": {}, "終盤": {}}
    for phase in result:
        mask = (tsumo_phase_labels == phase).values
        y_phase = y[mask]
        for col in target_cols:
            col_s = f"{col}{side_suffix}"
            if col_s not in paired.columns:
                continue
            x_phase = paired[col_s].astype(float).values[mask]
            r, _, _ = spearman_safe(x_phase, y_phase)
            result[phase][col] = r
    return result


# ========================================================================
# future_gain 計算
# ========================================================================

def compute_future_gain(df: pd.DataFrame, k_sec: int) -> pd.DataFrame:
    """
    各行について K 秒後の ojama_net_balance_raw を検索し future_gain を計算する。
    同一 (video_id, game_idx, side) 内のみ (試合境界を跨がない)。
    """
    col_name = f"future_gain_{k_sec}s"
    df = df.copy().reset_index(drop=True)
    df[col_name] = np.nan
    for _, grp in df.groupby(["video_id", "game_idx", "side"]):
        grp_sorted = grp.sort_values("t_sec")
        t_arr = grp_sorted["t_sec"].values.astype(float)
        net_arr = grp_sorted["ojama_net_balance_raw"].values.astype(float)
        orig_idx = grp_sorted.index.values
        for i in range(len(t_arr)):
            target_t = t_arr[i] + k_sec
            diffs = np.abs(t_arr - target_t)
            best = int(np.argmin(diffs))
            if diffs[best] <= LOOKAHEAD_TOLERANCE:
                df.at[orig_idx[i], col_name] = net_arr[best] - net_arr[i]
    return df


# ========================================================================
# 偏相関表の計算
# ========================================================================

class PartialCorrResult(NamedTuple):
    """偏相関 1 件の結果。"""
    col: str
    target: str         # "proxy" or "future_3s"
    raw_r: float        # 生 Spearman r
    raw_p: float
    partial_r: float    # 偏 Spearman r (board_color_puyo_total_raw 制御後)
    partial_p: float
    n: int
    is_fire_power: bool


def compute_partial_correlations(
    paired: pd.DataFrame,
    proxy: pd.Series,
    target_cols: list[str],
    future_gain_col: str | None,
) -> list[PartialCorrResult]:
    """
    各指標について生 Spearman r と偏 Spearman r (ぷよ数制御後) を計算する。
    """
    control_col = f"{PARTIAL_CONTROL_COL}_1p"
    if control_col not in paired.columns:
        print(f"[警告] 制御変数列 {control_col} が見つかりません。偏相関をスキップ。")
        return []

    z = paired[control_col].astype(float).values
    results: list[PartialCorrResult] = []

    # 目的変数リスト: proxy と future_gain
    targets: list[tuple[str, np.ndarray]] = [("proxy", proxy.values.astype(float))]
    fg_col_1p = f"{future_gain_col}_1p" if future_gain_col else None
    if fg_col_1p and fg_col_1p in paired.columns:
        targets.append(("future_3s", paired[fg_col_1p].astype(float).values))

    for tgt_name, y in targets:
        for col in target_cols:
            col_1p = f"{col}_1p"
            if col_1p not in paired.columns:
                continue
            x = paired[col_1p].astype(float).values
            raw_r, raw_p, n_raw = spearman_safe(x, y)
            part_r, part_p, _ = partial_spearman(x, y, z)
            results.append(PartialCorrResult(
                col=col, target=tgt_name,
                raw_r=raw_r, raw_p=raw_p,
                partial_r=part_r, partial_p=part_p,
                n=n_raw,
                is_fire_power=(col in FIRE_POWER_COLS),
            ))

    return results


# ========================================================================
# FDR 補正
# ========================================================================

def apply_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR 補正。有意フラグ (bool 配列) を返す。"""
    n = len(p_values)
    ps = np.where(np.isnan(p_values), 1.0, p_values)
    sorted_idx = np.argsort(ps)
    sorted_ps = ps[sorted_idx]
    thresholds = (np.arange(1, n + 1) / n) * FDR_ALPHA
    significant = np.zeros(n, dtype=bool)
    for k in range(n - 1, -1, -1):
        if sorted_ps[k] <= thresholds[k]:
            significant[sorted_idx[: k + 1]] = True
            break
    return significant


# ========================================================================
# フォーマットユーティリティ
# ========================================================================

def _fmt(v: float, width: int = 7) -> str:
    """float を +0.000 形式にフォーマット。nan は n/a。"""
    if np.isnan(v):
        return "  n/a ".rjust(width)
    return f"{v:+.3f}".rjust(width)


def _fmt_p(p: float) -> str:
    """p 値の短縮表示。"""
    if np.isnan(p):
        return "     n/a"
    if p < 0.001:
        return "  <.001"
    return f"{p:8.4f}"


# ========================================================================
# レポート出力
# ========================================================================

def print_report(
    paired: pd.DataFrame,
    proxy: pd.Series,
    puyo_phase_labels: pd.Series,
    tsumo_phase_labels: pd.Series,
    phase_counts: dict[str, int],
    proxy_puyo_corrs: dict[str, dict[str, float]],
    proxy_tsumo_corrs: dict[str, dict[str, float]],
    future_puyo_corrs: dict[str, dict[str, float]],
    future_tsumo_corrs: dict[str, dict[str, float]],
    partial_results: list[PartialCorrResult],
    pair_stats: dict,
    n_videos: int,
) -> None:
    """全分析結果をコンソールに出力。"""

    all_target_cols = FIRE_POWER_COLS + CONTROL_COLS

    print()
    print("=" * 95)
    print("  指標 v2 Phase2 -- 盤面ぷよ数位相 + 偏相関分析")
    print("=" * 95)
    print(f"  動画数: {n_videos}  ペア行数: {pair_stats['paired_rows']} / 1P行 {pair_stats['total_1p_rows']}")
    print(f"  ペア成立率: {pair_stats['pair_rate']:.1%}  t_diff中央値: {pair_stats['t_diff_median']:.2f}秒")
    print()

    # -- 位相サンプル数 --
    print("  ---- 盤面ぷよ数位相 サンプル数 (ペアリング後) ----")
    print(f"  {'位相':<10} {'件数':>6}  {'定義'}")
    print("  " + "-" * 60)
    defs = {
        "opening": f"max(手数1P,手数2P) <= {OPENING_TSUMO_MAX}  (最優先)",
        "序盤":    f"max_puyo <= {EARLY_PUYO_MAX}  (土台組み込み)",
        "中盤":    f"{EARLY_PUYO_MAX} < max_puyo <= {LATE_PUYO_MIN}",
        "終盤":    f"max_puyo > {LATE_PUYO_MIN}  (サンプル少・要注意)",
    }
    for phase in ["opening", "序盤", "中盤", "終盤"]:
        print(f"  {phase:<10} {phase_counts[phase]:>6}  {defs[phase]}")
    print()

    # ====================================================================
    # 1. proxy 相関: 盤面ぷよ数位相 vs 手数三分位 対比表
    # ====================================================================
    print("  ====================================================================")
    print("  [1] proxy 相関 -- 盤面ぷよ数位相 vs 手数三分位 (Spearman r)")
    print("  ====================================================================")
    print(f"  proxy = {PROXY_W_OJAMA}*z(ojama_net_balance_raw) + {PROXY_W_DEATH}*z(death_margin_diff)")
    print()
    hdr = (f"  {'指標':<30} {'種':>3}  "
           f"{'opening':>8} {'序盤P':>8} {'中盤P':>8} {'終盤P':>8}  "
           f"{'序盤T':>8} {'中盤T':>8} {'終盤T':>8}")
    print(hdr)
    print("  " + "-" * 96)
    print("  ※P=ぷよ数位相、T=手数三分位")
    for group_label, cols_group in [("火力系", FIRE_POWER_COLS), ("対照系", CONTROL_COLS)]:
        print(f"    [{group_label}]")
        for col in cols_group:
            typ = "火" if col in FIRE_POWER_COLS else "対"
            p_op = proxy_puyo_corrs["opening"].get(col, float("nan"))
            p_e  = proxy_puyo_corrs["序盤"].get(col, float("nan"))
            p_m  = proxy_puyo_corrs["中盤"].get(col, float("nan"))
            p_l  = proxy_puyo_corrs["終盤"].get(col, float("nan"))
            t_e  = proxy_tsumo_corrs["序盤"].get(col, float("nan"))
            t_m  = proxy_tsumo_corrs["中盤"].get(col, float("nan"))
            t_l  = proxy_tsumo_corrs["終盤"].get(col, float("nan"))
            print(
                f"  {col:<30} {typ:>3}  "
                f"{_fmt(p_op):>8} {_fmt(p_e):>8} {_fmt(p_m):>8} {_fmt(p_l):>8}  "
                f"{_fmt(t_e):>8} {_fmt(t_m):>8} {_fmt(t_l):>8}"
            )
    print()

    # ====================================================================
    # 2. future_gain(K=3s) 相関: 盤面ぷよ数位相 vs 手数三分位 対比表
    # ====================================================================
    print("  ====================================================================")
    print(f"  [2] future_gain({LOOKAHEAD_K_SEC}s) 相関 -- 盤面ぷよ数位相 vs 手数三分位 (Spearman r)")
    print("  ====================================================================")
    print(f"  future_gain_{LOOKAHEAD_K_SEC}s = {LOOKAHEAD_K_SEC}秒後の ojama_net_balance_raw 差分")
    print()
    print(hdr)
    print("  " + "-" * 96)
    print("  ※P=ぷよ数位相、T=手数三分位")
    for group_label, cols_group in [("火力系", FIRE_POWER_COLS), ("対照系", CONTROL_COLS)]:
        print(f"    [{group_label}]")
        for col in cols_group:
            typ = "火" if col in FIRE_POWER_COLS else "対"
            p_op = future_puyo_corrs["opening"].get(col, float("nan"))
            p_e  = future_puyo_corrs["序盤"].get(col, float("nan"))
            p_m  = future_puyo_corrs["中盤"].get(col, float("nan"))
            p_l  = future_puyo_corrs["終盤"].get(col, float("nan"))
            t_e  = future_tsumo_corrs["序盤"].get(col, float("nan"))
            t_m  = future_tsumo_corrs["中盤"].get(col, float("nan"))
            t_l  = future_tsumo_corrs["終盤"].get(col, float("nan"))
            print(
                f"  {col:<30} {typ:>3}  "
                f"{_fmt(p_op):>8} {_fmt(p_e):>8} {_fmt(p_m):>8} {_fmt(p_l):>8}  "
                f"{_fmt(t_e):>8} {_fmt(t_m):>8} {_fmt(t_l):>8}"
            )
    print()

    # ====================================================================
    # 3. 偏相関表: 生相関 vs ぷよ数制御後の偏相関
    # ====================================================================
    print("  ====================================================================")
    print("  [3] 偏相関表 -- board_color_puyo_total_raw を制御変数として除去後")
    print("  ====================================================================")
    print(f"  制御変数: {PARTIAL_CONTROL_COL} (1P側の値)")
    print("  偏 Spearman r = rank 変換後 OLS 残差の Pearson r (近似)")
    print("  Δr = 偏相関 - 生相関 (マイナスなら盤面密度で説明される部分が大きい)")
    print("  FDR: Benjamini-Hochberg alpha=0.05 (偏相関 p 値に適用)")
    print()

    # FDR 補正 (偏相関の p 値に適用)
    partial_ps = np.array([r.partial_p if not np.isnan(r.partial_p) else 1.0 for r in partial_results])
    partial_fdr_flags = apply_fdr(partial_ps)

    for tgt_label in ["proxy", "future_3s"]:
        tgt_display = "proxy" if tgt_label == "proxy" else f"future_gain_{LOOKAHEAD_K_SEC}s"
        print(f"  -- 目的変数: {tgt_display} --")
        print(
            f"  {'指標':<30} {'種':>3}  "
            f"{'生Sp-r':>8} {'生p':>8}  "
            f"{'偏Sp-r':>8} {'偏p':>8} {'FDR':>5}  "
            f"{'Δr':>8}  n"
        )
        print("  " + "-" * 90)

        tgt_items = [
            (r, partial_fdr_flags[i])
            for i, r in enumerate(partial_results)
            if r.target == tgt_label
        ]
        for group_label, cols_group in [("火力系", FIRE_POWER_COLS), ("対照系", CONTROL_COLS)]:
            print(f"    [{group_label}]")
            # |偏相関| 降順
            group_items = sorted(
                [(r, sig) for r, sig in tgt_items if r.col in cols_group],
                key=lambda x: abs(x[0].partial_r) if not np.isnan(x[0].partial_r) else 0.0,
                reverse=True,
            )
            for r, sig in group_items:
                typ = "火" if r.is_fire_power else "対"
                delta_r = (r.partial_r - r.raw_r
                           if not (np.isnan(r.partial_r) or np.isnan(r.raw_r))
                           else float("nan"))
                fdr_mark = "  *  " if sig else "     "
                print(
                    f"  {r.col:<30} {typ:>3}  "
                    f"{_fmt(r.raw_r):>8} {_fmt_p(r.raw_p):>8}  "
                    f"{_fmt(r.partial_r):>8} {_fmt_p(r.partial_p):>8} {fdr_mark:>5}  "
                    f"{_fmt(delta_r):>8}  {r.n}"
                )
        print()

    # ====================================================================
    # 注意事項
    # ====================================================================
    print("  ====================================================================")
    print("  注意事項")
    print("  ====================================================================")
    print(f"  - 終盤 (max_puyo > {LATE_PUYO_MIN}) は {phase_counts['終盤']} 件と少ない。")
    print("    上級者試合では積み上げ前に連鎖発火が多いため。終盤結果は参考値扱い推奨。")
    print(f"  - opening は max(tsumo_1p, tsumo_2p) <= {OPENING_TSUMO_MAX} で判定。")
    print("    連鎖直後にぷよが減っても手数は戻らないため「本物の序盤」のみ選択される。")
    print("  - 偏相関は rank 変換後 OLS 残差の線形近似。")
    print("    board_color_puyo_total_raw と対象指標が非線形関係の場合は過小評価あり。")
    print(f"  - future_gain_{LOOKAHEAD_K_SEC}s は試合終了直前 (K秒後が存在しない) を除外。")
    print("  - LOOV (Leave-One-Video-Out) は本スクリプトでは省略。汎化性は analyze_indicator_proxy.py 参照。")
    print()


# ========================================================================
# CSV 保存
# ========================================================================

def save_csv(
    proxy_puyo_corrs: dict[str, dict[str, float]],
    proxy_tsumo_corrs: dict[str, dict[str, float]],
    future_puyo_corrs: dict[str, dict[str, float]],
    future_tsumo_corrs: dict[str, dict[str, float]],
    partial_results: list[PartialCorrResult],
    out_path: str,
) -> None:
    """分析結果を CSV に保存する。"""
    rows: list[dict] = []
    all_cols = FIRE_POWER_COLS + CONTROL_COLS
    for col in all_cols:
        row: dict = {"col": col, "is_fire_power": col in FIRE_POWER_COLS}
        for phase in ["opening", "序盤", "中盤", "終盤"]:
            row[f"proxy_puyo_{phase}"] = proxy_puyo_corrs[phase].get(col, float("nan"))
            row[f"future_puyo_{phase}"] = future_puyo_corrs[phase].get(col, float("nan"))
        for phase in ["序盤", "中盤", "終盤"]:
            row[f"proxy_tsumo_{phase}"] = proxy_tsumo_corrs[phase].get(col, float("nan"))
            row[f"future_tsumo_{phase}"] = future_tsumo_corrs[phase].get(col, float("nan"))
        rows.append(row)

    phase_df = pd.DataFrame(rows)
    partial_df = pd.DataFrame([r._asdict() for r in partial_results])

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("# phase_correlations\n")
        phase_df.to_csv(f, index=False)
        f.write("\n# partial_correlations\n")
        partial_df.to_csv(f, index=False)

    print(f"  CSV 保存: {out_path}")


# ========================================================================
# main
# ========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="指標 v2 盤面ぷよ数位相 + 偏相関分析")
    parser.add_argument(
        "--study", default="data/indicators_v2/study",
        help="study ディレクトリ (デフォルト: data/indicators_v2/study)",
    )
    parser.add_argument(
        "--max-tdiff", type=float, default=DEFAULT_MAX_TDIFF,
        help=f"ペアリング最大時刻差 秒 (デフォルト: {DEFAULT_MAX_TDIFF})",
    )
    parser.add_argument(
        "--out", default=None,
        help="結果 CSV 出力パス (省略時はコンソールのみ)",
    )
    args = parser.parse_args()

    print(f"[analyze_indicator_phase2] study={args.study} max_tdiff={args.max_tdiff}秒")

    # ---- データ読み込み ----
    df_raw = load_study_csvs(args.study)
    n_videos = int(df_raw["video_id"].nunique())

    # 数値化 (メタ列以外)
    meta_cols = {"video_id", "game_idx", "side", "reach_fire_power_source", "chain_duration_source"}
    for col in df_raw.columns:
        if col not in meta_cols:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    # ---- future_gain 計算 (生データに対して) ----
    print(f"\n[future_gain_{LOOKAHEAD_K_SEC}s 計算中...]")
    df_with_fg = compute_future_gain(df_raw, LOOKAHEAD_K_SEC)
    fg_col = f"future_gain_{LOOKAHEAD_K_SEC}s"
    n_excluded = int(df_with_fg[fg_col].isna().sum())
    n_valid = len(df_with_fg) - n_excluded
    print(f"  K={LOOKAHEAD_K_SEC}秒: 有効行={n_valid}/{len(df_with_fg)} (除外={n_excluded})")

    # ---- ペアリング ----
    print("\n[ペアリング中...]")
    paired, pair_stats = pair_sides(df_with_fg, args.max_tdiff)
    if len(paired) == 0:
        print("[ERROR] ペアが 1 件も成立しませんでした。--max-tdiff を増やしてください。")
        sys.exit(1)
    print(f"  ペア数: {len(paired)}")

    # ---- proxy 計算 ----
    proxy = compute_proxy(paired)

    # ---- 位相ラベル付与 ----
    print("\n[位相ラベル計算中...]")
    puyo_phase_labels = assign_puyo_phase(paired)
    tsumo_phase_labels = assign_tsumo_phase(paired)
    phase_counts = count_phase_samples(puyo_phase_labels)
    print("  盤面ぷよ数位相 サンプル数:")
    for phase, cnt in phase_counts.items():
        print(f"    {phase}: {cnt}")

    all_target_cols = FIRE_POWER_COLS + CONTROL_COLS

    # ---- proxy 相関 (盤面ぷよ数位相) ----
    print("\n[proxy 相関 (盤面ぷよ数位相) 計算中...]")
    proxy_puyo_corrs = compute_phase_correlations_by_puyo(
        paired, proxy.values, puyo_phase_labels, all_target_cols,
    )

    # ---- proxy 相関 (手数三分位位相) ----
    print("[proxy 相関 (手数三分位) 計算中...]")
    proxy_tsumo_corrs = compute_phase_correlations_by_tsumo(
        paired, proxy.values, tsumo_phase_labels, all_target_cols,
    )

    # ---- future_gain 相関 (盤面ぷよ数位相) ----
    print("[future_gain 相関 (盤面ぷよ数位相) 計算中...]")
    fg_col_1p = f"{fg_col}_1p"
    if fg_col_1p not in paired.columns:
        print(f"[警告] {fg_col_1p} がペアリング済みDataFrameに見つかりません。")
        future_y = np.full(len(paired), float("nan"))
    else:
        future_y = paired[fg_col_1p].astype(float).values
    future_puyo_corrs = compute_phase_correlations_by_puyo(
        paired, future_y, puyo_phase_labels, all_target_cols,
    )

    # ---- future_gain 相関 (手数三分位位相) ----
    print("[future_gain 相関 (手数三分位) 計算中...]")
    future_tsumo_corrs = compute_phase_correlations_by_tsumo(
        paired, future_y, tsumo_phase_labels, all_target_cols,
    )

    # ---- 偏相関 ----
    print("\n[偏相関 (盤面ぷよ数制御) 計算中...]")
    partial_results = compute_partial_correlations(
        paired, proxy, all_target_cols, fg_col,
    )
    print(f"  偏相関計算完了: {len(partial_results)} 件")

    # ---- レポート出力 ----
    print_report(
        paired=paired,
        proxy=proxy,
        puyo_phase_labels=puyo_phase_labels,
        tsumo_phase_labels=tsumo_phase_labels,
        phase_counts=phase_counts,
        proxy_puyo_corrs=proxy_puyo_corrs,
        proxy_tsumo_corrs=proxy_tsumo_corrs,
        future_puyo_corrs=future_puyo_corrs,
        future_tsumo_corrs=future_tsumo_corrs,
        partial_results=partial_results,
        pair_stats=pair_stats,
        n_videos=n_videos,
    )

    if args.out:
        save_csv(
            proxy_puyo_corrs, proxy_tsumo_corrs,
            future_puyo_corrs, future_tsumo_corrs,
            partial_results, args.out,
        )


if __name__ == "__main__":
    main()
