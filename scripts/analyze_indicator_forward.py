"""指標 v2 -- 火力系指標の前向き (forward-looking) 予測力分析。"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import NamedTuple
import numpy as np
import pandas as pd
from scipy import stats

LOOKAHEAD_K_LIST: list[int] = [3, 5, 10]
LOOKAHEAD_TOLERANCE: float = 1.5
FDR_ALPHA: float = 0.05
TSUMO_EARLY_MAX: int = 13
TSUMO_MID_MAX: int = 40

FIRE_POWER_COLS: list[str] = [
    'current_max_chain',
    'immediate_fire_power',
    'reach_fire_power',
    'chain_efficiency',
    'second_chain_potential',
    'min_puyos_to_ignite',
    'conn_pair_count',
    'conn_triple_count',
    'conn_max_group_size',
]
CONTROL_COLS: list[str] = [
    'board_ojama_count',
    'max_column_height',
    'board_puyo_total',
    'death_margin',
    'margin_time_rate',
]
ALL_TARGET_COLS: list[str] = FIRE_POWER_COLS + CONTROL_COLS

def load_study_csvs(study_dir: str) -> pd.DataFrame:
    """study ディレクトリの全 CSV を結合して返す (corr_20v.csv は除外)。"""
    paths = sorted(Path(study_dir).glob('*.csv'))
    paths = [p for p in paths if p.name != 'corr_20v.csv']
    if not paths:
        raise FileNotFoundError(f'CSV が見つかりません: {study_dir}')
    dfs: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        df['_source_file'] = p.name
        dfs.append(df)
        print(f'  読み込み: {p.name}  {df.shape[0]} 行')
    combined = pd.concat(dfs, ignore_index=True)
    print(f'  合計: {combined.shape[0]} 行, {combined.shape[1]} 列')
    return combined


def compute_future_gain(df: pd.DataFrame, k_sec: int, tol: float = LOOKAHEAD_TOLERANCE) -> pd.DataFrame:
    """
    各行について K秒後の ojama_net_balance_raw を検索し future_gain を計算する。
    同一 (video_id, game_idx, side) 内のみ (試合境界を跨がない)。
    K秒後スナップショットが tol 以内に存在しない行は future_gain=NaN。
    """
    col_name = f'future_gain_{k_sec}s'
    df = df.copy().reset_index(drop=True)
    df[col_name] = np.nan
    for _, grp in df.groupby(['video_id', 'game_idx', 'side']):
        grp_sorted = grp.sort_values('t_sec')
        t_arr = grp_sorted['t_sec'].values.astype(float)
        net_arr = grp_sorted['ojama_net_balance_raw'].values.astype(float)
        orig_idx = grp_sorted.index.values
        for i in range(len(t_arr)):
            t0 = t_arr[i]
            target_t = t0 + k_sec
            diffs = np.abs(t_arr - target_t)
            best = int(np.argmin(diffs))
            if diffs[best] <= tol:
                df.at[orig_idx[i], col_name] = net_arr[best] - net_arr[i]
    return df


def add_all_future_gains(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    """全 K についての future_gain を df に追加し、除外行数を返す。"""
    excluded_counts: dict[int, int] = {}
    for k in LOOKAHEAD_K_LIST:
        df = compute_future_gain(df, k)
        col = f'future_gain_{k}s'
        n_excluded = int(df[col].isna().sum())
        excluded_counts[k] = n_excluded
        print(f'  K={k}秒: 有効行={len(df) - n_excluded}, 除外行={n_excluded}')
    return df, excluded_counts


class CorrResult(NamedTuple):
    """1指標 x 1K x 1位相 の相関結果。"""
    col: str
    k_sec: int
    phase: str
    spearman_r: float
    spearman_p: float
    n: int
    loov_r_mean: float
    loov_r_std: float
    is_fire_power: bool


def spearman_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman r, p を安全に計算。サンプル不足は (nan, nan)。"""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 5:
        return float('nan'), float('nan')
    r, p = stats.spearmanr(x[mask], y[mask], nan_policy='omit')
    return float(r), float(p)
def compute_forward_correlations(df: pd.DataFrame, target_cols: list[str], k_list: list[int]) -> list[CorrResult]:
    """各指標 x K x 位相 について Spearman 相関 + LOOV を計算。"""
    videos = df['video_id'].unique()
    results: list[CorrResult] = []
    phase_masks: dict[str, np.ndarray] = {
        '全体': np.ones(len(df), dtype=bool),
        '序盤': (df['tsumo'].values <= TSUMO_EARLY_MAX),
        '中盤': (
            (df['tsumo'].values > TSUMO_EARLY_MAX)
            & (df['tsumo'].values <= TSUMO_MID_MAX)
        ),
        '終盤': (df['tsumo'].values > TSUMO_MID_MAX),
    }
    for k in k_list:
        future_col = f'future_gain_{k}s'
        if future_col not in df.columns: continue
        y_all = df[future_col].values.astype(float)
        for col in target_cols:
            if col not in df.columns: continue
            x_all = df[col].values.astype(float)
            for phase, pmask in phase_masks.items():
                x = x_all[pmask]
                y = y_all[pmask]
                sp_r, sp_p = spearman_safe(x, y)
                n_valid = int((~np.isnan(x) & ~np.isnan(y)).sum())
                loov_rs: list[float] = []
                for v in videos:
                    loov_pmask = pmask & (df['video_id'].values != v)
                    r_v, _ = spearman_safe(x_all[loov_pmask], y_all[loov_pmask])
                    if not np.isnan(r_v): loov_rs.append(r_v)
                loov_mean = float(np.mean(loov_rs)) if loov_rs else float('nan')
                loov_std = float(np.std(loov_rs, ddof=1)) if len(loov_rs) > 1 else float('nan')
                results.append(CorrResult(col=col, k_sec=k, phase=phase,
                    spearman_r=sp_r, spearman_p=sp_p, n=n_valid,
                    loov_r_mean=loov_mean, loov_r_std=loov_std,
                    is_fire_power=(col in FIRE_POWER_COLS)))
    return results


def apply_fdr(results: list[CorrResult]) -> list[tuple[CorrResult, bool]]:
    """Benjamini-Hochberg FDR 補正 (全体 phase のみで算出)。"""
    global_results = [r for r in results if r.phase == '全体']
    ps = np.array([r.spearman_p if not np.isnan(r.spearman_p) else 1.0 for r in global_results])
    n = len(ps)
    sorted_idx = np.argsort(ps)
    sorted_ps = ps[sorted_idx]
    thresholds = (np.arange(1, n + 1) / n) * FDR_ALPHA
    significant_flags = np.zeros(n, dtype=bool)
    for ki in range(n - 1, -1, -1):
        if sorted_ps[ki] <= thresholds[ki]:
            significant_flags[sorted_idx[:ki + 1]] = True
            break
    sig_map: dict[tuple[str, int], bool] = {}
    for i, r in enumerate(global_results):
        sig_map[(r.col, r.k_sec)] = bool(significant_flags[i])
    return [(r, sig_map.get((r.col, r.k_sec), False)) for r in results]


def fmt_r(v: float) -> str:
    """float を +0.000 形式でフォーマット。nan は n/a。"""
    if np.isnan(v): return '   n/a'
    return f'{v:+.3f}'

def print_report(results_fdr, df, excluded_counts, n_videos):
    """分析結果をコンソールに出力。"""
    print()
    print('============================================================================================')
    print('  指標 v2 -- 火力系指標の前向き予測力分析 (future_gain 相関)')
    print('============================================================================================')
    print(f'  動画数: {n_videos}  スナップショット総数: {len(df)}')
    print(f'  K={LOOKAHEAD_K_LIST}秒 先の ojama_net_balance_raw 変化量 (future_gain) を目的変数')
    print(f'  許容誤差: +-{LOOKAHEAD_TOLERANCE}秒以内に K秒後スナップショットが存在する行のみ使用')
    print()
    for k, excl in excluded_counts.items():
        n_valid = len(df) - excl
        print(f'  K={k}秒: 有効={n_valid}/{len(df)} (除外={excl}, {excl/len(df)*100:.1f}%)')
    print()
    print(f'  手数区切り: 序盤 tsumo<={TSUMO_EARLY_MAX} 中盤 <={TSUMO_MID_MAX} 終盤 >{TSUMO_MID_MAX}')
    print(f'  LOOV: Leave-One-Video-Out ({n_videos} 本)')
    print(f'  FDR: Benjamini-Hochberg alpha={FDR_ALPHA} (全体 phase で算出)')
    print()
    for k in LOOKAHEAD_K_LIST:
        print(f'  ---- K={k}秒先 future_gain との Spearman r ----')
        HA, HB, HC, HD, HE, HF, HG, HH, HI = '指標', '種別', '全体-r', 'LOOV-r', 'LOOV-std', 'FDR', '序盤', '中盤', '終盤'
        print(f'  {HA:<30} {HB:>6} {HC:>8} {HD:>8} {HE:>9} {HF:>5} {HG:>7} {HH:>7} {HI:>7}  n')
        print('  ' + '-' * 90)
        k_global = {r.col: (r, sig) for r, sig in results_fdr if r.k_sec == k and r.phase == '全体'}
        k_early = {r.col: r.spearman_r for r, _ in results_fdr if r.k_sec == k and r.phase == '序盤'}
        k_mid   = {r.col: r.spearman_r for r, _ in results_fdr if r.k_sec == k and r.phase == '中盤'}
        k_late  = {r.col: r.spearman_r for r, _ in results_fdr if r.k_sec == k and r.phase == '終盤'}
        for group_label, cols_group in [('火力系', FIRE_POWER_COLS), ('対照系', CONTROL_COLS)]:
            print(f'    [{group_label}]')
            rows_s = sorted([(col, *k_global[col]) for col in cols_group if col in k_global],
                key=lambda x: abs(x[1].loov_r_mean) if not np.isnan(x[1].loov_r_mean) else 0.0, reverse=True)
            for col, r, sig in rows_s:
                typ = '火力' if r.is_fire_power else '対照'
                fm = '  *  ' if sig else '     '
                er = k_early.get(col, float('nan'))
                mr = k_mid.get(col, float('nan'))
                lr = k_late.get(col, float('nan'))
                print(f'  {col:<30} {typ:>6} {fmt_r(r.spearman_r):>8} {fmt_r(r.loov_r_mean):>8} {fmt_r(r.loov_r_std):>9} {fm:>5} {fmt_r(er):>7} {fmt_r(mr):>7} {fmt_r(lr):>7}  {r.n}')
        print()
    CA, CB = '指標', '種別'
    print('  ---- 横断比較: K別 LOOV-r (全体 phase) / 平均|r| 降順 ----')
    hdr2 = f'  {CA:<30} {CB:>6}'
    for k in LOOKAHEAD_K_LIST: hdr2 += f' K={k}s  '
    hdr2 += ' 平均|r|'
    print(hdr2)
    print('  ' + '-' * 72)
    all_rows: list[tuple] = []
    for cols_group, lbl in [(FIRE_POWER_COLS, '火力'), (CONTROL_COLS, '対照')]:
        for col in cols_group:
            kv: list[float] = []
            la: list[float] = []
            for k in LOOKAHEAD_K_LIST:
                m = next((r for r, _ in results_fdr if r.col == col and r.k_sec == k and r.phase == '全体'), None)
                v = m.loov_r_mean if m is not None else float('nan')
                kv.append(v)
                if not np.isnan(v): la.append(abs(v))
            aa = float(np.mean(la)) if la else float('nan')
            all_rows.append((col, lbl, kv, aa))
    all_rows_sorted = sorted(all_rows, key=lambda x: x[3] if not np.isnan(x[3]) else 0.0, reverse=True)
    for col, lbl, kv, aa in all_rows_sorted:
        pr = [f'  {col:<30} {lbl:>6}']
        for v in kv: pr.append(f' {fmt_r(v):>7}')
        pr.append(f' {aa:+.3f}' if not np.isnan(aa) else '    n/a')
        print(''.join(pr))
    print()
    print('  注意事項:')
    print('  - future_gain = K秒後の ojama_net_balance_raw 差分 (+=送出優勢が伸びた)。')
    print('  - ojama_net_balance_raw は 0 が多く (序盤は均衡)、')
    print('    future_gain の分散が小さい -> 相関が低くなる傾向あり。')
    print('  - OCR 誤差: ojama_net_balance_raw には +-数個のノイズが含まれる可能性がある。')
    print('  - 循環参照なし: future_gain は現在値でなく将来差分。')
    print('    ただし火力系と future_gain が共に盤面密度に依存する交絡は残る。')
    print()

def save_csv(results_fdr: list[tuple[CorrResult, bool]], out_path: str) -> None:
    """分析結果を CSV に保存。"""
    rows = []
    for r, sig in results_fdr:
        rows.append({'col': r.col, 'k_sec': r.k_sec, 'phase': r.phase,
            'spearman_r': r.spearman_r, 'spearman_p': r.spearman_p, 'n': r.n,
            'loov_r_mean': r.loov_r_mean, 'loov_r_std': r.loov_r_std,
            'fdr_significant': sig, 'is_fire_power': r.is_fire_power})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'  CSV 保存: {out_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='指標 v2 前向き予測力分析')
    parser.add_argument('--study', default='data/indicators_v2/study')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    print(f'[analyze_indicator_forward] study={args.study}')
    df = load_study_csvs(args.study)
    n_videos = int(df['video_id'].nunique())
    for col in ALL_TARGET_COLS + ['ojama_net_balance_raw', 't_sec', 'tsumo']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print('\n[future_gain 計算中...]')
    df, excluded_counts = add_all_future_gains(df)
    print('\n[Spearman 相関計算中... (LOOV 込み)]')
    raw_results = compute_forward_correlations(df, ALL_TARGET_COLS, LOOKAHEAD_K_LIST)
    results_fdr = apply_fdr(raw_results)
    print_report(results_fdr, df, excluded_counts, n_videos)
    if args.out:
        save_csv(results_fdr, args.out)


if __name__ == '__main__':
    main()
