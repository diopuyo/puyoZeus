# -*- coding: utf-8 -*-
"""
STABLE持続ゲート (--enable-stable-persistence-gate) による行数61%減の内訳診断。
既存npz(新旧)の読み取り専用。src/配下・collect_boards_lean.pyは一切変更しない。

user指摘1: 窓長0.25秒が長すぎ、1手サイクルの完全静止区間より長いのでは
  → 旧データの「次の行が現れるまでの間隔(=その盤面状態の寿命)」を使い、
    0.25/0.15/0.10秒未満だった状態が理論上何%あったかを試算する
    (生diff系列はnpzに保存されていないため、正確な再計算ではなく
    上限見積り=optimisticな試算である旨を明記する)。

user指摘2: 相手バースト中(CHAIN)は自分の盤面が静止していても
  画素差分が出てゲートに弾かれるのでは
  → 相手側 chain_trigger_sec 起点の「相手バースト窓」を定義し、
    窓内/窓外で自分側の行密度(行数/秒)が旧→新でどれだけ落ちたかを層別。
"""
import numpy as np
from pathlib import Path

OLD_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-11")
NEW_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-18")

OPP_BURST_WINDOWS_SEC = [3.0, 5.0]  # 相手バースト窓の仮定候補 (CHAIN_ANIM_PER_STEP_SEC=0.4*chain数の目安)
INTERVAL_THRESHOLDS = [0.25, 0.15, 0.10]


def load(path):
    return dict(np.load(path, allow_pickle=True))


def merge_windows(events, win):
    """[(c, c+win), ...] を時刻順にマージして重なりを解消する。"""
    if len(events) == 0:
        return []
    spans = sorted((c, c + win) for c in events)
    merged = [spans[0]]
    for s, e in spans[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def time_in_windows(t_array, windows):
    """t_array の各要素が windows のいずれかに入るかの bool 配列。"""
    if not windows:
        return np.zeros(len(t_array), dtype=bool)
    inside = np.zeros(len(t_array), dtype=bool)
    for s, e in windows:
        inside |= (t_array >= s) & (t_array < e)
    return inside


def total_window_duration(windows, t_min, t_max):
    """windows を [t_min, t_max] にクリップした合計秒数。"""
    total = 0.0
    for s, e in windows:
        s2, e2 = max(s, t_min), min(e, t_max)
        if e2 > s2:
            total += e2 - s2
    return total


def analyze_interval_thresholds(old):
    """旧データの盤面状態の寿命(次の行までの間隔)分布から、
    窓長を短縮したら理論上何%が「窓長だけで届かない」かを試算する
    (楽観的上限。実際にはバースト等の追加diffでさらに悪化しうる)。
    """
    side = old["side"]; gi = old["game_idx"]; t = old["t_sec"]
    intervals = []
    for s in np.unique(side):
        m = side == s
        tt = t[m]; ggi = gi[m]
        order = np.argsort(tt)
        tt = tt[order]; ggi = ggi[order]
        for i in range(1, len(tt)):
            if ggi[i] == ggi[i - 1]:
                intervals.append(tt[i] - tt[i - 1])
    intervals = np.array(intervals)
    result = {}
    for th in INTERVAL_THRESHOLDS:
        result[th] = float((intervals < th).mean()) if len(intervals) else float("nan")
    return result, intervals


def analyze_burst_stratification(video_id, old, new):
    """相手バースト窓(内/外)で、自分側の行密度(行/秒)が旧→新でどう変化したかを層別。"""
    out = {}
    for win_sec in OPP_BURST_WINDOWS_SEC:
        rows_for_win = []
        for side in ["1P", "2P"]:
            opp = "2P" if side == "1P" else "1P"
            # 相手のchain_trigger_secは新データを基準に使う(現行state machine相当)
            opp_new = new["chain_trigger_sec"][new["side"] == opp]
            opp_new = opp_new[np.isfinite(opp_new)]
            windows = merge_windows(opp_new.tolist(), win_sec)

            own_old_t = old["t_sec"][old["side"] == side]
            own_new_t = new["t_sec"][new["side"] == side]
            if len(own_old_t) == 0 or len(own_new_t) == 0:
                continue
            t_min = min(own_old_t.min(), own_new_t.min())
            t_max = max(own_old_t.max(), own_new_t.max())

            t_chain = total_window_duration(windows, t_min, t_max)
            t_far = (t_max - t_min) - t_chain
            if t_chain <= 0 or t_far <= 0:
                continue

            in_win_old = time_in_windows(own_old_t, windows)
            in_win_new = time_in_windows(own_new_t, windows)

            n_chain_old = int(in_win_old.sum()); n_far_old = int((~in_win_old).sum())
            n_chain_new = int(in_win_new.sum()); n_far_new = int((~in_win_new).sum())

            rate_chain_old = n_chain_old / t_chain
            rate_far_old = n_far_old / t_far
            rate_chain_new = n_chain_new / t_chain
            rate_far_new = n_far_new / t_far

            ratio_chain = rate_chain_new / rate_chain_old if rate_chain_old > 0 else float("nan")
            ratio_far = rate_far_new / rate_far_old if rate_far_old > 0 else float("nan")

            rows_for_win.append(dict(
                side=side, win_sec=win_sec,
                t_chain=round(t_chain, 1), t_far=round(t_far, 1),
                n_chain_old=n_chain_old, n_chain_new=n_chain_new,
                n_far_old=n_far_old, n_far_new=n_far_new,
                ratio_chain=ratio_chain, ratio_far=ratio_far,
            ))
        out[win_sec] = rows_for_win
    return out


if __name__ == "__main__":
    new_files = sorted(p.stem for p in NEW_DIR.glob("*.npz"))
    all_interval_stats = []
    all_burst_stats = {w: [] for w in OPP_BURST_WINDOWS_SEC}

    for vid in new_files:
        old_p = OLD_DIR / f"{vid}.npz"
        new_p = NEW_DIR / f"{vid}.npz"
        if not old_p.exists():
            continue
        old = load(old_p)
        new = load(new_p)

        th_result, intervals = analyze_interval_thresholds(old)
        all_interval_stats.append((vid, th_result))

        burst_result = analyze_burst_stratification(vid, old, new)
        for w, rows in burst_result.items():
            all_burst_stats[w].extend(rows)

        print(f"\n=== {vid} ===")
        print(f"  旧データ盤面寿命(次行までの間隔) < 閾値 の割合(理論的上限見積り):")
        for th, frac in th_result.items():
            print(f"    < {th}s: {frac*100:.1f}%")
        for w, rows in burst_result.items():
            for r in rows:
                print(f"  [burst_win={w}s side={r['side']}] "
                      f"chain窓内: old={r['n_chain_old']} new={r['n_chain_new']} "
                      f"(t={r['t_chain']}s) ratio={r['ratio_chain']:.3f} | "
                      f"chain窓外: old={r['n_far_old']} new={r['n_far_new']} "
                      f"(t={r['t_far']}s) ratio={r['ratio_far']:.3f}")

    print("\n\n=== まとめ: 窓長感度試算 (旧データ盤面寿命分布ベース、楽観的上限) ===")
    for th in INTERVAL_THRESHOLDS:
        vals = [d[th] for _, d in all_interval_stats]
        print(f"  < {th}s の割合: mean={100*np.mean(vals):.1f}% "
              f"min={100*np.min(vals):.1f}% max={100*np.max(vals):.1f}%")

    print("\n=== まとめ: 相手バースト窓 内/外 の行密度比 (new/old) ===")
    for w in OPP_BURST_WINDOWS_SEC:
        rows = all_burst_stats[w]
        rc = [r["ratio_chain"] for r in rows if np.isfinite(r["ratio_chain"])]
        rf = [r["ratio_far"] for r in rows if np.isfinite(r["ratio_far"])]
        print(f"  win={w}s  n_sides={len(rows)}")
        print(f"    窓内(相手バースト中)  ratio mean={np.mean(rc):.3f} median={np.median(rc):.3f} n={len(rc)}")
        print(f"    窓外(相手バースト外)  ratio mean={np.mean(rf):.3f} median={np.median(rf):.3f} n={len(rf)}")
