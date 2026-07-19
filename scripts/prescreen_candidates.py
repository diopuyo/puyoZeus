from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

N_FOLDS: int = 5
MAX_TDIFF: float = 1.0
TSUMO_EARLY_MAX: float = 0.33
TSUMO_LATE_MIN: float = 0.67
ROLLING_WINDOWS: list[int] = [3, 5, 10]
EPS: float = 1.0
BASELINE_MID_AUC: float = 0.56
BASELINE_LATE_AUC: float = 0.68
PHASE_ALL: str = "all"
PHASE_EARLY: str = "early"
PHASE_MID: str = "mid"
PHASE_LATE: str = "late"


def load_and_pair(labeled_path: str) -> pd.DataFrame:
    df = pd.read_csv(labeled_path)
    df = df[df["won"].notna()].copy()
    df["won"] = df["won"].astype(int)
    print(f"[load] won 付き行: {len(df)}")
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
            if diffs[idx_min] > MAX_TDIFF:
                continue
            r2 = g2.iloc[idx_min]
            w1, w2 = float(r1["won"]), float(r2["won"])
            if abs(w1 + w2 - 1.0) > 0.01:
                continue
            row: dict = {}
            for col in r1.index:
                row[f"{col}_1p"] = r1[col]
            for col in g2.columns:
                row[f"{col}_2p"] = r2[col]
            row["t_diff"] = diffs[idx_min]
            rows.append(row)
    paired = pd.DataFrame(rows)
    print(f"[pair] ペア成立: {len(paired)} 行")
    return paired


def _s(p: pd.DataFrame, col: str, side: str) -> pd.Series:
    key = f"{col}_{side}"
    if key in p.columns:
        return p[key].astype(float)
    return pd.Series(np.nan, index=p.index)


def build_relational_features(paired: pd.DataFrame) -> pd.DataFrame:
    feats: dict[str, pd.Series] = {}
    feats["lethality_1p"] = _s(paired, "ojama_forecast_raw", "1p") / (
        _s(paired, "absorption_capacity_raw", "1p").clip(lower=EPS))
    feats["lethality_2p"] = _s(paired, "ojama_forecast_raw", "2p") / (
        _s(paired, "absorption_capacity_raw", "2p").clip(lower=EPS))
    feats["lethality_diff"] = feats["lethality_1p"] - feats["lethality_2p"]

    feats["reach_vs_opp_abs_1p"] = _s(paired, "reach_fire_power_raw", "1p") / (
        _s(paired, "absorption_capacity_raw", "2p").clip(lower=EPS))
    feats["reach_vs_opp_abs_2p"] = _s(paired, "reach_fire_power_raw", "2p") / (
        _s(paired, "absorption_capacity_raw", "1p").clip(lower=EPS))
    feats["reach_vs_opp_abs_diff"] = (
        feats["reach_vs_opp_abs_1p"] - feats["reach_vs_opp_abs_2p"])

    feats["death_margin_diff"] = (
        _s(paired, "death_margin_raw", "1p") - _s(paired, "death_margin_raw", "2p"))
    feats["death_margin_ratio"] = _s(paired, "death_margin_raw", "1p") / (
        _s(paired, "death_margin_raw", "2p").clip(lower=EPS))

    feats["chain_ratio"] = _s(paired, "current_max_chain_raw", "1p") / (
        _s(paired, "current_max_chain_raw", "2p").clip(lower=EPS))
    feats["chain_diff"] = (
        _s(paired, "current_max_chain_raw", "1p")
        - _s(paired, "current_max_chain_raw", "2p"))

    feats["reach_ratio"] = _s(paired, "reach_fire_power_raw", "1p") / (
        _s(paired, "reach_fire_power_raw", "2p").clip(lower=EPS))
    feats["reach_diff"] = (
        _s(paired, "reach_fire_power_raw", "1p")
        - _s(paired, "reach_fire_power_raw", "2p"))

    feats["board_color_diff"] = (
        _s(paired, "board_color_puyo_total_raw", "1p")
        - _s(paired, "board_color_puyo_total_raw", "2p"))

    c1 = paired.get("conn_pair_count_1p", pd.Series(dtype=float))
    c2 = paired.get("conn_pair_count_2p", pd.Series(dtype=float))
    if not c1.empty and not c2.empty:
        feats["conn_pair_diff"] = c1.astype(float) - c2.astype(float)

    feats["ojama_forecast_diff"] = (
        _s(paired, "ojama_forecast_raw", "1p")
        - _s(paired, "ojama_forecast_raw", "2p"))
    feats["ojama_net_diff"] = (
        _s(paired, "ojama_net_balance_raw", "1p")
        - _s(paired, "ojama_net_balance_raw", "2p"))

    return pd.DataFrame(feats, index=paired.index)


def _rolling_delta(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df.groupby(
        ["video_id", "game_idx"], group_keys=False
    )[col].transform(
        lambda s: s.diff().rolling(window, min_periods=1).mean()
    )


def _nearest_join(
    src: pd.DataFrame,
    delta_col: str,
    vid_arr: np.ndarray,
    t_arr: np.ndarray,
) -> pd.Series:
    out = np.full(len(t_arr), np.nan)
    for vid in np.unique(vid_arr):
        sub = src[src["video_id"] == vid]
        if sub.empty:
            continue
        idxs = np.where(vid_arr == vid)[0]
        t_s = sub["t_sec"].values
        t_q = t_arr[idxs]
        nearest = np.abs(t_s[:, None] - t_q[None, :]).argmin(axis=0)
        out[idxs] = sub[delta_col].values[nearest]
    return pd.Series(out)


def build_temporal_features(
    raw_1p: pd.DataFrame,
    raw_2p: pd.DataFrame,
    vid_arr: np.ndarray,
    t_arr: np.ndarray,
) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    raw_cols = {
        "board_color_puyo_total_raw": "build_speed",
        "current_max_chain_raw": "chain_growth",
        "ojama_forecast_raw": "ojama_trend",
        "death_margin_raw": "death_approach",
    }
    dm2_on_1p = _nearest_join(
        raw_2p, "death_margin_raw",
        raw_1p["video_id"].values, raw_1p["t_sec"].values)
    raw_1p_aug = raw_1p.copy()
    raw_1p_aug["_dm_diff"] = (
        raw_1p["death_margin_raw"].values - dm2_on_1p.values)
    for window in ROLLING_WINDOWS:
        for raw_col, feat_base in raw_cols.items():
            src1 = raw_1p.copy()
            src1["_d"] = _rolling_delta(raw_1p, raw_col, window)
            s1 = _nearest_join(src1, "_d", vid_arr, t_arr)
            src2 = raw_2p.copy()
            src2["_d"] = _rolling_delta(raw_2p, raw_col, window)
            s2 = _nearest_join(src2, "_d", vid_arr, t_arr)
            result[f"{feat_base}_w{window}_1p"] = s1
            result[f"{feat_base}_w{window}_2p"] = s2
            result[f"{feat_base}_w{window}_diff"] = s1 - s2
        src_mom = raw_1p_aug.copy()
        src_mom["_d"] = _rolling_delta(raw_1p_aug, "_dm_diff", window)
        result[f"momentum_w{window}"] = _nearest_join(
            src_mom, "_d", vid_arr, t_arr)
        if window >= 5:
            raw_1p_aug["_sign"] = np.sign(
                raw_1p_aug["_dm_diff"].ffill().fillna(0))
            raw_1p_aug["_flip"] = (
                raw_1p_aug.groupby(
                    ["video_id", "game_idx"], group_keys=False
                )["_sign"].transform(
                    lambda s: s.diff().abs().gt(0)
                    .rolling(window, min_periods=1).sum()
                )
            )
            result[f"sign_flip_w{window}"] = _nearest_join(
                raw_1p_aug, "_flip", vid_arr, t_arr)
    return result


def univariate_auc(
    x: pd.Series,
    y: pd.Series,
    groups: pd.Series,
    mask: pd.Series,
) -> float:
    idx = mask & y.notna() & x.notna()
    if idx.sum() < 30:
        return float("nan")
    xi = x[idx].values
    yi = y[idx].values
    gi = groups[idx].values
    n_uniq = len(np.unique(gi))
    gkf = GroupKFold(n_splits=min(N_FOLDS, n_uniq))
    oof = np.full(len(xi), np.nan)
    for tr_idx, va_idx in gkf.split(xi, yi, gi):
        std = float(np.std(xi[tr_idx]))
        if std < 1e-9:
            oof[va_idx] = 0.5
            continue
        corr = float(np.corrcoef(xi[tr_idx], yi[tr_idx])[0, 1])
        sign = np.sign(corr) if corr != 0 else 1.0
        oof[va_idx] = xi[va_idx] * sign
    valid = ~np.isnan(oof)
    if valid.sum() < 10:
        return float("nan")
    try:
        return float(roc_auc_score(yi[valid], oof[valid]))
    except Exception:
        return float("nan")


def eval_candidate(
    x: pd.Series,
    y: pd.Series,
    groups: pd.Series,
    phase_masks: dict[str, pd.Series],
) -> dict[str, float]:
    return {ph: univariate_auc(x, y, groups, m) for ph, m in phase_masks.items()}



def print_summary(df: pd.DataFrame) -> None:
    hdr = 'header_placeholder'
    sep = '-' * 70
    def _f(v: object) -> str:
        try:
            return '%6.4f' % float(v)
        except Exception:
            return '   NaN'
    def fmt_row(r: pd.Series) -> str:
        return '%-42s %s %s %s %s %4s' % (
            str(r['candidate']),
            _f(r.get('all')), _f(r.get('early')),
            _f(r.get('mid')), _f(r.get('late')),
            str(r.get('window', '')),
        )
    mk, lk = 'mid', 'late'
    mid_s = df.dropna(subset=[mk]).sort_values(mk, ascending=False)
    late_s = df.dropna(subset=[lk]).sort_values(lk, ascending=False)
    print('')
    print('## 中盤 AUC 上位 10')
    print(hdr.replace('header_placeholder', '候補                                       全体   序盤   中盤   終盤    窓'))
    print(sep)
    for _, r in mid_s.head(10).iterrows():
        print(fmt_row(r))
    print('')
    print('## 終盤 AUC 上位 10')
    print(hdr.replace('header_placeholder', '候補                                       全体   序盤   中盤   終盤    窓'))
    print(sep)
    for _, r in late_s.head(10).iterrows():
        print(fmt_row(r))
    bm, bl = BASELINE_MID_AUC, BASELINE_LATE_AUC
    print('')
    over_mid = df[df[mk].fillna(0) > bm].sort_values(mk, ascending=False)
    over_late = df[df[lk].fillna(0) > bl].sort_values(lk, ascending=False)
    print('## 既存超過 (中盤>%s / 終盤>%s)' % (bm, bl))
    if len(over_mid):
        print('  [中盤超過 %d 候補]' % len(over_mid))
        for _, r in over_mid.iterrows():
            try:
                lv = '%.4f' % float(r.get(lk))
            except Exception:
                lv = 'NaN'
            print('    %s: mid=%.4f  late=%s' % (r['candidate'], r[mk], lv))
    else:
        print('  [中盤超過] なし')
    if len(over_late):
        print('  [終盤超過 %d 候補]' % len(over_late))
        for _, r in over_late.iterrows():
            try:
                mv = '%.4f' % float(r.get(mk))
            except Exception:
                mv = 'NaN'
            print('    %s: late=%.4f  mid=%s' % (r['candidate'], r[lk], mv))
    else:
        print('  [終盤超過] なし')


def main() -> None:
    parser = argparse.ArgumentParser(description='指標候補の事前スクリーニング')
    parser.add_argument('--labeled',
                        default='data/indicators_v2/study/labeled_win.csv')
    parser.add_argument('--out',
                        default='data/indicators_v2/prescreen_auc.csv')
    args = parser.parse_args()
    print('=== prescreen_candidates 開始 ===')
    print('  入力: ' + args.labeled)
    print('  出力: ' + args.out)
    paired = load_and_pair(args.labeled)
    df_all = pd.read_csv(args.labeled)
    df_all = df_all[df_all['won'].notna()].copy()
    raw_1p = df_all[df_all['side'] == '1P'].sort_values(
        ['video_id', 'game_idx', 't_sec']).reset_index(drop=True)
    raw_2p = df_all[df_all['side'] == '2P'].sort_values(
        ['video_id', 'game_idx', 't_sec']).reset_index(drop=True)
    won_cols = [c for c in paired.columns if 'won' in c and c.endswith('_1p')]
    if not won_cols:
        print('[ERROR] won_1p 列が見つかりません', file=sys.stderr)
        sys.exit(1)
    y = paired[won_cols[0]].astype(float)
    groups = paired['video_id_1p']
    tcr = paired['tsumo_count_rate_1p'].astype(float)
    phase_masks = {
        PHASE_ALL:   pd.Series(True, index=paired.index),
        PHASE_EARLY: tcr <= TSUMO_EARLY_MAX,
        PHASE_MID:   (tcr > TSUMO_EARLY_MAX) & (tcr <= TSUMO_LATE_MIN),
        PHASE_LATE:  tcr > TSUMO_LATE_MIN,
    }
    for ph, m in phase_masks.items():
        print('  位相 %s: %d 行' % (ph, m.sum()))
    auc_rows: list[dict] = []
    print('')
    print('[eval] 関係的候補...')
    rel = build_relational_features(paired)
    for col in rel.columns:
        aucs = eval_candidate(rel[col], y, groups, phase_masks)
        auc_rows.append({'candidate': col, 'category': 'relational',
                         'window': None, **aucs})
    print('  完了: %d 候補' % len(rel.columns))
    print('')
    print('[eval] 時系列候補...')
    vid_arr = paired['video_id_1p'].values
    t_arr = paired['t_sec_1p'].values
    temporal = build_temporal_features(raw_1p, raw_2p, vid_arr, t_arr)
    for name, series in temporal.items():
        w = next((ww for ww in ROLLING_WINDOWS if '_w%d' % ww in name), None)
        cat = ('temporal_rel'
               if ('momentum' in name or 'sign_flip' in name
                   or name.endswith('_diff'))
               else 'temporal')
        aucs = eval_candidate(series, y, groups, phase_masks)
        auc_rows.append({'candidate': name, 'category': cat,
                         'window': w, **aucs})
    print('  完了: %d 候補' % len(temporal))
    result_df = pd.DataFrame(auc_rows)
    result_df = result_df.sort_values(PHASE_MID, ascending=False, na_position='last')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.out, index=False)
    print('')
    print('[save] %s (%d 候補)' % (args.out, len(result_df)))
    print_summary(result_df)
    print('')
    print('=== 完了 ===')


if __name__ == '__main__':
    main()
