"""指標 v2 -- 優勢proxy vs 各指標 Spearman/Pearson 相関ランキング。

## proxy 定義 (暫定 w1=0.7, w2=0.3)
    proxy = 0.7 * z(ojama_net_balance_raw) + 0.3 * z(death_margin_raw - opp_death_margin_raw)
- ojama_net_balance_raw: +=送り優勢 (このサイド視点, collect_indicators_v2 で符号付き)
- death_margin_raw: このサイドの窒息余裕 (0=即死, 12=安全)
- w1/w2 は暫定。重みの変更はスクリプト冒頭定数 PROXY_W_OJAMA / PROXY_W_DEATH で。

## 交絡・過学習対策
- LOOV (Leave-One-Video-Out): 1動画をvalidationにして残りで相関算出し、平均/std を報告
- 多重比較補正: Benjamini-Hochberg FDR (alpha=0.05)
- 手数三分位別 (序盤/中盤/終盤) で相関を分離報告

## 使い方
    python -m scripts.analyze_indicator_proxy --study data/indicators_v2/study
    python -m scripts.analyze_indicator_proxy --study data/indicators_v2/study --max-tdiff 1.0

## 出力
    - コンソールにテキスト表 (リダイレクト可)
    - --out FILE を指定すると CSV で保存

再利用: 動画 CSV を study ディレクトリに追加後、同コマンドで再実行可。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

# proxy 重み (暫定。変更はここだけ)
PROXY_W_OJAMA: float = 0.7
PROXY_W_DEATH: float = 0.3

# ペアリング最大時刻差 (秒)
DEFAULT_MAX_TDIFF: float = 1.0

# FDR 有意水準
FDR_ALPHA: float = 0.05

# 手数三分位の境界
TSUMO_EARLY_RATIO: float = 0.33
TSUMO_LATE_RATIO: float = 0.67

# proxy の構成要素列 (自明に高相関のため本命ランキングから除外)
PROXY_COMPONENTS: frozenset[str] = frozenset([
    "ojama_net_balance", "ojama_net_balance_raw",
    "ojama_forecast", "ojama_forecast_raw",
    "death_margin", "death_margin_raw",
    "death_margin_neighbor", "death_margin_neighbor_raw",
])

# 非数値・メタ・raw 列 (分析対象外)
SKIP_COLS: frozenset[str] = frozenset([
    "video_id", "game_idx", "t_sec", "frame", "tsumo", "side",
    "reach_fire_power_source", "chain_duration_source",
    "tsumo_count_raw", "board_puyo_total_raw", "board_color_puyo_total_raw",
    "margin_time_rate_raw", "max_column_height_raw", "column_bumpiness_raw",
    "death_margin_raw", "death_margin_neighbor_raw",
    "current_max_chain_raw", "immediate_fire_power_raw", "reach_fire_power_raw",
    "reach_fire_power_max_chain",
    "chain_efficiency_raw", "min_puyos_to_ignite_raw",
    "second_chain_potential_raw",
    "ojama_net_balance_raw", "ojama_forecast_raw", "board_ojama_count_raw",
    "chain_duration_sec",
    "dig_resistance_raw", "absorption_capacity_raw",
])


def load_study_csvs(study_dir: str) -> pd.DataFrame:
    """study ディレクトリの全 CSV を結合して返す。"""
    paths = sorted(Path(study_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"CSV が見つかりません: {study_dir}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        dfs.append(df)
        print(f"  読み込み: {p.name}  {df.shape[0]} 行")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  合計: {combined.shape[0]} 行, {combined.shape[1]} 列")
    return combined


def pair_sides(df: pd.DataFrame, max_tdiff: float) -> tuple[pd.DataFrame, dict]:
    """1P/2P を (video_id, game_idx) 内で時刻最近傍マッチ。"""
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
        "t_diff_mean": paired["t_diff"].mean() if len(paired) > 0 else float("nan"),
        "t_diff_median": paired["t_diff"].median() if len(paired) > 0 else float("nan"),
        "max_tdiff_threshold": max_tdiff,
    }
    return paired, stats_info


def compute_proxy(paired: pd.DataFrame) -> pd.Series:
    """優勢 proxy を算出して Series で返す。"""
    ojama_raw = paired["ojama_net_balance_raw_1p"].astype(float)
    dm_diff = (
        paired["death_margin_raw_1p"].astype(float)
        - paired["death_margin_raw_2p"].astype(float)
    )

    def zscore(s: pd.Series) -> pd.Series:
        std = s.std(ddof=1)
        if std < 1e-9:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / std

    return PROXY_W_OJAMA * zscore(ojama_raw) + PROXY_W_DEATH * zscore(dm_diff)


class CorrResult(NamedTuple):
    """1指標の相関結果。"""
    col: str
    spearman_r: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    n: int
    loov_spearman_mean: float
    loov_spearman_std: float
    is_proxy_component: bool


def _corr_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Spearman / Pearson の (r, p) を返す。サンプル不足なら nan。"""
    if len(x) < 5:
        return float("nan"), float("nan"), float("nan"), float("nan")
    sp_r, sp_p = stats.spearmanr(x, y, nan_policy="omit")
    pe_r, pe_p = stats.pearsonr(x, y)
    return float(sp_r), float(sp_p), float(pe_r), float(pe_p)


def compute_correlations(
    paired: pd.DataFrame,
    proxy: pd.Series,
    indicator_cols: list[str],
) -> list[CorrResult]:
    """各指標について全体相関 + LOOV 相関を計算。"""
    videos = paired["video_id_1p"].unique()
    results: list[CorrResult] = []
    for col in indicator_cols:
        col_1p = f"{col}_1p"
        if col_1p not in paired.columns:
            continue
        x = paired[col_1p].astype(float).values
        y = proxy.values
        mask = ~(np.isnan(x) | np.isnan(y))
        sp_r, sp_p, pe_r, pe_p = _corr_pair(x[mask], y[mask])
        loov_rs: list[float] = []
        for v in videos:
            loov_mask = (paired["video_id_1p"] != v).values & mask
            if loov_mask.sum() < 5:
                continue
            r_v, _, _, _ = _corr_pair(x[loov_mask], y[loov_mask])
            if not np.isnan(r_v):
                loov_rs.append(r_v)
        loov_mean = float(np.mean(loov_rs)) if loov_rs else float("nan")
        loov_std = float(np.std(loov_rs, ddof=1)) if len(loov_rs) > 1 else float("nan")
        results.append(CorrResult(
            col=col, spearman_r=sp_r, spearman_p=sp_p,
            pearson_r=pe_r, pearson_p=pe_p, n=int(mask.sum()),
            loov_spearman_mean=loov_mean, loov_spearman_std=loov_std,
            is_proxy_component=(col in PROXY_COMPONENTS),
        ))
    return results


def apply_fdr(results: list[CorrResult]) -> list[tuple[CorrResult, bool]]:
    """Benjamini-Hochberg FDR 補正で有意フラグを付与。"""
    ps = np.array([r.spearman_p if not np.isnan(r.spearman_p) else 1.0 for r in results])
    n = len(ps)
    sorted_idx = np.argsort(ps)
    sorted_ps = ps[sorted_idx]
    thresholds = (np.arange(1, n + 1) / n) * FDR_ALPHA
    significant = np.zeros(n, dtype=bool)
    for k in range(n - 1, -1, -1):
        if sorted_ps[k] <= thresholds[k]:
            significant[sorted_idx[: k + 1]] = True
            break
    return [(r, bool(significant[i])) for i, r in enumerate(results)]


def compute_phase_correlations(
    paired: pd.DataFrame,
    proxy: pd.Series,
    indicator_cols: list[str],
    tsumo_q33: float,
    tsumo_q67: float,
) -> dict[str, dict[str, float]]:
    """序盤/中盤/終盤別に Spearman r を返す。"""
    phase_masks = {
        "序盤": (paired["tsumo_1p"] <= tsumo_q33).values,
        "中盤": ((paired["tsumo_1p"] > tsumo_q33) & (paired["tsumo_1p"] <= tsumo_q67)).values,
        "終盤": (paired["tsumo_1p"] > tsumo_q67).values,
    }
    phase_results: dict[str, dict[str, float]] = {}
    for phase, mask in phase_masks.items():
        phase_proxy = proxy.values[mask]
        phase_dict: dict[str, float] = {}
        for col in indicator_cols:
            col_1p = f"{col}_1p"
            if col_1p not in paired.columns:
                continue
            x = paired[col_1p].astype(float).values[mask]
            nm = ~(np.isnan(x) | np.isnan(phase_proxy))
            if nm.sum() < 5:
                phase_dict[col] = float("nan")
                continue
            r, _ = stats.spearmanr(x[nm], phase_proxy[nm], nan_policy="omit")
            phase_dict[col] = float(r)
        phase_results[phase] = phase_dict
    return phase_results


def _fmt_r(v: float) -> str:
    if np.isnan(v):
        return "  n/a "
    return f"{v:+.3f}"


def print_report(
    results_fdr: list[tuple[CorrResult, bool]],
    phase_results: dict[str, dict[str, float]],
    pair_stats: dict,
    tsumo_q33: float,
    tsumo_q67: float,
    n_csvs: int,
) -> None:
    """分析結果をコンソールに出力。"""
    print()
    print("=" * 80)
    print("  指標 v2 -- 優勢 proxy 相関ランキング  (暫定: 3本プレビュー)")
    print("=" * 80)
    print(f"  CSV 本数: {n_csvs}  (全10本揃い次第 --study 再実行で更新)")
    print(
        f"  ペア: {pair_stats['paired_rows']} / 1P行 {pair_stats['total_1p_rows']}"
        f"  (成立率 {pair_stats['pair_rate']:.1%},"
        f" t_diff 中央値 {pair_stats['t_diff_median']:.2f}秒,"
        f" 閾値 {pair_stats['max_tdiff_threshold']}秒)"
    )
    print(f"  FDR alpha={FDR_ALPHA}  (Benjamini-Hochberg)")
    print()
    print("  proxy 定義:")
    print(f"    proxy = {PROXY_W_OJAMA} * z(ojama_net_balance_raw)")
    print(f"          + {PROXY_W_DEATH} * z(death_margin_raw_1p - death_margin_raw_2p)")
    print("    ※ お邪魔net収支・窒息余裕は proxy 構成要素 -> 参考列に分離")
    print()
    print(
        f"  手数三分位: 序盤 tsumo<={tsumo_q33:.0f},"
        f" 中盤 {tsumo_q33:.0f}< tsumo <={tsumo_q67:.0f},"
        f" 終盤 tsumo>{tsumo_q67:.0f}"
    )
    print()

    main_res = [(r, sig) for r, sig in results_fdr if not r.is_proxy_component]
    main_sorted = sorted(
        main_res,
        key=lambda x: abs(x[0].loov_spearman_mean)
        if not np.isnan(x[0].loov_spearman_mean) else 0.0,
        reverse=True,
    )
    print("  -- 本命指標ランキング (proxy 構成要素を除く) --")
    print(
        f"  {'rank':>4} {'指標':<30} {'Sp-r(全)':>9} {'LOOV-r':>9} {'LOOV-std':>9}"
        f" {'FDR':>5} {'序盤':>7} {'中盤':>7} {'終盤':>7}  n"
    )
    print("  " + "-" * 88)
    for rank, (r, sig) in enumerate(main_sorted, 1):
        early = _fmt_r(phase_results["序盤"].get(r.col, float("nan")))
        mid = _fmt_r(phase_results["中盤"].get(r.col, float("nan")))
        late = _fmt_r(phase_results["終盤"].get(r.col, float("nan")))
        fdr_mark = "  *  " if sig else "     "
        print(
            f"  {rank:>4} {r.col:<30}"
            f" {_fmt_r(r.spearman_r):>9}"
            f" {_fmt_r(r.loov_spearman_mean):>9}"
            f" {_fmt_r(r.loov_spearman_std):>9}"
            f" {fdr_mark:>5}"
            f" {early:>7} {mid:>7} {late:>7}"
            f"  {r.n}"
        )

    print()
    print("  -- 参考: proxy 構成要素 --")
    comp_res = [(r, sig) for r, sig in results_fdr if r.is_proxy_component]
    comp_sorted = sorted(
        comp_res,
        key=lambda x: abs(x[0].spearman_r) if not np.isnan(x[0].spearman_r) else 0.0,
        reverse=True,
    )
    print(f"  {'指標':<30} {'Sp-r(全)':>9} {'LOOV-r':>9}")
    print("  " + "-" * 52)
    for r, _ in comp_sorted:
        print(f"  {r.col:<30} {_fmt_r(r.spearman_r):>9} {_fmt_r(r.loov_spearman_mean):>9}")

    print()
    print("  注意事項:")
    print("  - 3本 (v29/v30/v31) の暫定結果。動画数少で LOOV-std が大きく結論は仮。")
    print("  - ペア成立率 55%: 1P/2P の STABLE タイミング非同期が主因。")
    print("  - proxy は勝敗ラベルの代理信号。proxy と相関 != 最終勝率と相関。")
    print("  - FDR * は 3本データでの有意性。10本揃い次第再評価が必要。")
    print()


def save_csv(
    results_fdr: list[tuple[CorrResult, bool]],
    phase_results: dict[str, dict[str, float]],
    out_path: str,
) -> None:
    """分析結果を CSV に保存。"""
    rows = []
    for r, sig in results_fdr:
        row = {
            "col": r.col,
            "spearman_r": r.spearman_r,
            "spearman_p": r.spearman_p,
            "pearson_r": r.pearson_r,
            "pearson_p": r.pearson_p,
            "n": r.n,
            "loov_spearman_mean": r.loov_spearman_mean,
            "loov_spearman_std": r.loov_spearman_std,
            "fdr_significant": sig,
            "is_proxy_component": r.is_proxy_component,
            "spearman_early": phase_results["序盤"].get(r.col, float("nan")),
            "spearman_mid": phase_results["中盤"].get(r.col, float("nan")),
            "spearman_late": phase_results["終盤"].get(r.col, float("nan")),
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  CSV 保存: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="指標 v2 proxy 相関分析")
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
    print(f"[analyze_indicator_proxy] study={args.study} max_tdiff={args.max_tdiff}秒")

    df = load_study_csvs(args.study)
    n_csvs = df["video_id"].nunique()

    paired, pair_stats = pair_sides(df, args.max_tdiff)
    if len(paired) == 0:
        print("[ERROR] ペアが 1 件も成立しませんでした。--max-tdiff を増やしてください。")
        sys.exit(1)

    proxy = compute_proxy(paired)

    p1_all = df[df["side"] == "1P"]["tsumo"]
    tsumo_q33 = float(p1_all.quantile(TSUMO_EARLY_RATIO))
    tsumo_q67 = float(p1_all.quantile(TSUMO_LATE_RATIO))

    all_score_cols = [
        c for c in df.columns
        if c not in SKIP_COLS
        and not c.endswith("_raw")
        and not c.endswith("_source")
        and c not in ("reach_fire_power_max_chain",)
    ]
    numeric_cols = [c for c in all_score_cols if pd.api.types.is_numeric_dtype(df[c])]

    raw_results = compute_correlations(paired, proxy, numeric_cols)
    results_fdr = apply_fdr(raw_results)
    phase_results = compute_phase_correlations(paired, proxy, numeric_cols, tsumo_q33, tsumo_q67)
    print_report(results_fdr, phase_results, pair_stats, tsumo_q33, tsumo_q67, n_csvs)

    if args.out:
        save_csv(results_fdr, phase_results, args.out)


if __name__ == "__main__":
    main()
