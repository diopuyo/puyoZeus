"""公平条件での新旧再比較 準備スクリプト (2026-08-19)。

背景: 新CSVは --exclude-match-end-locked (match_end_locked==1 or
post_match_lockdown_active==1 の行除外) 適用済みだが、旧CSV/旧npzには
該当列が存在しない (旧npzは chain_mechanism までの14列、再ビルド不能)。
そこで新npzの locked 時間窓 (t_sec区間) を旧CSVへ転写して同等フィルタを
事後適用する。

手順:
  1. 整合検証: 旧npz vs 新npz の game_idx 開始時刻 (side別) を最近傍対応させ、
     |Δt| の分布を出す (再DL内容ドリフト = feedback_redownload_content_drift
     の検知)。ズレが大きい動画は転写不可として警告。
  2. locked窓の抽出: 新npz side別・t_sec昇順で locked の連続 run を区間化。
     境界は隣接 unlocked 行との中点まで拡張 (エッジバイアス低減)。
  3. won欠損窓の抽出 (変種b用): unlocked かつ won NaN の連続 run を同様に区間化。
  4. 旧CSVへ適用して2変種を出力:
     a. old_lockedfilt : locked窓のみ除外 (主比較)
     b. old_lockedfilt_wonsync : locked窓 + won欠損窓 も除外 (感度分析)

出力:
  data/verify/retrain_subset42_2026-08-19/old_lockedfilt/labeled_win_old_lockedfilt.csv
  data/verify/retrain_subset42_2026-08-19/old_lockedfilt_wonsync/labeled_win_old_lockedfilt_wonsync.csv
  logs/diag_fair_recompare_build_2026-08-19.log (stdout をリダイレクトして使う)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OLD_NPZ_DIR = ROOT / "data/indicators_v2/boards_lean_phase_l_2026-08-11"
NEW_NPZ_DIR = ROOT / "data/indicators_v2/boards_lean_subset50_2026-08-19"
OLD_CSV = ROOT / "data/verify/retrain_subset42_2026-08-19/old/labeled_win_old.csv"
OUT_A = ROOT / "data/verify/retrain_subset42_2026-08-19/old_lockedfilt/labeled_win_old_lockedfilt.csv"
OUT_B = ROOT / "data/verify/retrain_subset42_2026-08-19/old_lockedfilt_wonsync/labeled_win_old_lockedfilt_wonsync.csv"

TARGET_IDS = [
    "29", "36", "39", "52", "c100", "c101", "c102", "c103", "c104", "c105",
    "c106", "c107", "c108", "c109", "c11", "c110", "c111", "c112", "c113",
    "c114", "c115", "c116", "c117", "c118", "c119", "c125", "c126", "c127",
    "c128", "c129", "c13", "c130", "c131", "c132", "c133", "c134", "c135",
    "c136", "c137", "c96s1", "c96s2", "c96s3",
]

# 整合検証: 最近傍対応の許容 (秒) と、転写可とみなす中央値|Δt|の上限 (秒)
ALIGN_MATCH_TOL_SEC: float = 10.0
ALIGN_MEDIAN_ABS_MAX_SEC: float = 2.0


def _game_start_times(t: np.ndarray, g: np.ndarray) -> np.ndarray:
    """t_sec昇順に並べた上で、game_idx が変わる行の t_sec を返す。"""
    order = np.argsort(t, kind="stable")
    ts, gs = t[order], g[order]
    if len(ts) == 0:
        return np.array([])
    starts = [ts[0]]
    for i in range(1, len(ts)):
        if gs[i] != gs[i - 1]:
            starts.append(ts[i])
    return np.asarray(starts, dtype=float)


def _runs_to_intervals(t_sorted: np.ndarray, flag_sorted: np.ndarray) -> list[tuple[float, float]]:
    """t_sec昇順の行列から flag==True の連続 run を [start,end] 区間にする。

    区間端は隣接する flag==False 行との中点まで拡張する (新npzは疎サンプリング
    のため、run の生端点だけだと旧CSVの密な行が境界ギャップに取り残される)。
    """
    n = len(t_sorted)
    intervals: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if not flag_sorted[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flag_sorted[j + 1]:
            j += 1
        start = t_sorted[i] if i == 0 else 0.5 * (t_sorted[i - 1] + t_sorted[i])
        end = t_sorted[j] if j == n - 1 else 0.5 * (t_sorted[j] + t_sorted[j + 1])
        # run が動画末尾まで続く場合は無限大側へ開く
        if j == n - 1:
            end = float("inf")
        intervals.append((float(start), float(end)))
        i = j + 1
    return intervals


def _mask_in_intervals(t: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(len(t), dtype=bool)
    for s, e in intervals:
        m |= (t >= s) & (t <= e)
    return m


def main() -> int:
    # ---- 1. 整合検証 ----
    print("=== 1. 旧npz vs 新npz の時刻整合 (game_idx開始時刻の最近傍|Δt|) ===")
    align_bad: list[str] = []
    locked_iv: dict[tuple[str, str], list[tuple[float, float]]] = {}
    wonnan_iv: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for vid in TARGET_IDS:
        d_old = np.load(OLD_NPZ_DIR / f"{vid}.npz", allow_pickle=True)
        d_new = np.load(NEW_NPZ_DIR / f"{vid}.npz", allow_pickle=True)
        stats = []
        for side in ("1P", "2P"):
            mo = d_old["side"] == side
            locked_new = (d_new["match_end_locked"] == 1) | (d_new["post_match_lockdown_active"] == 1)
            mn = (d_new["side"] == side)
            # 開始時刻対応 (新側は unlocked のみで開始時刻を取る)
            so = _game_start_times(d_old["t_sec"][mo], d_old["game_idx"][mo])
            sn = _game_start_times(
                d_new["t_sec"][mn & ~locked_new], d_new["game_idx"][mn & ~locked_new])
            deltas = []
            for tstart in sn:
                if len(so) == 0:
                    continue
                k = int(np.argmin(np.abs(so - tstart)))
                if abs(so[k] - tstart) <= ALIGN_MATCH_TOL_SEC:
                    deltas.append(so[k] - tstart)
            stats.append((side, len(sn), len(deltas),
                          float(np.median(np.abs(deltas))) if deltas else float("nan")))
            # locked窓 / won欠損窓 (side別, t_sec昇順)
            tn = d_new["t_sec"][mn]
            order = np.argsort(tn, kind="stable")
            tns = tn[order].astype(float)
            lk = locked_new[mn][order]
            wn = np.isnan(d_new["won"][mn][order]) & ~lk
            locked_iv[(vid, side)] = _runs_to_intervals(tns, lk)
            wonnan_iv[(vid, side)] = _runs_to_intervals(tns, wn)
        med = np.nanmedian([s[3] for s in stats])
        matched = sum(s[2] for s in stats)
        total = sum(s[1] for s in stats)
        flag = ""
        if not np.isfinite(med) or med > ALIGN_MEDIAN_ABS_MAX_SEC or matched < total * 0.7:
            flag = "  <-- 整合不良 (転写不可候補)"
            align_bad.append(vid)
        print(f"  {vid:>6}: 新側境界{total}件中 対応{matched}件, median|Δt|={med:.3f}s{flag}")

    print(f"\n整合不良動画: {align_bad if align_bad else 'なし'}")

    # ---- 2. 旧CSVへ適用 ----
    print("\n=== 2. 旧CSVへの locked窓転写 ===")
    df = pd.read_csv(OLD_CSV)
    n0 = len(df)
    drop_locked = np.zeros(n0, dtype=bool)
    drop_wonnan = np.zeros(n0, dtype=bool)
    for (vid, side), ivs in locked_iv.items():
        m = (df["video_id"] == f"video_{vid}") & (df["side"] == side)
        t = df.loc[m, "t_sec"].to_numpy(dtype=float)
        drop_locked[np.where(m)[0]] |= _mask_in_intervals(t, ivs)
    for (vid, side), ivs in wonnan_iv.items():
        m = (df["video_id"] == f"video_{vid}") & (df["side"] == side)
        t = df.loc[m, "t_sec"].to_numpy(dtype=float)
        drop_wonnan[np.where(m)[0]] |= _mask_in_intervals(t, ivs)

    df_a = df[~drop_locked]
    df_b = df[~(drop_locked | drop_wonnan)]
    OUT_A.parent.mkdir(parents=True, exist_ok=True)
    OUT_B.parent.mkdir(parents=True, exist_ok=True)
    df_a.to_csv(OUT_A, index=False)
    df_b.to_csv(OUT_B, index=False)

    print(f"旧CSV               : {n0:,}行")
    print(f"locked窓除外        : -{int(drop_locked.sum()):,} ({drop_locked.sum()/n0:.1%})"
          f" -> a. {len(df_a):,}行 -> {OUT_A}")
    print(f"won欠損窓 追加除外   : -{int((drop_wonnan & ~drop_locked).sum()):,}"
          f" -> b. {len(df_b):,}行 -> {OUT_B}")
    print("参考: 新CSV(学習到達)=189,545行 / 新npz locked比率=16.5%")

    # per-video の除外率 (locked窓)
    print("\n=== per-video locked窓除外率 (旧CSV) ===")
    tmp = df.assign(_drop=drop_locked).groupby("video_id")["_drop"].agg(["sum", "size"])
    tmp["rate"] = tmp["sum"] / tmp["size"]
    print(tmp.sort_values("rate", ascending=False).head(12).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
