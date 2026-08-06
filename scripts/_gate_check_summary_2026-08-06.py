import json
from pathlib import Path

PER_COL_UNKNOWN_WARNING = 0.15
PER_COL_UNKNOWN_CRITICAL = 0.30
MIDGAME_COL_EMPTY_CRITICAL = 0.99
MIDGAME_COL_MIN_FRAMES = 30

data = json.loads(Path("data/verify/burst_guard_2026-08-05/_gate_check_2026-08-06_result.json").read_text())

print(f"n_videos={len(data)}")
print()
print("=== 0. カバレッジ (on_v2_full の anchor全長に対する時間カバー率) ===")
covs = sorted(data, key=lambda r: r["coverage_ratio"])
for r in covs:
    print(f"{r['video']:6s} on_t_max={r['on_t_range'][1]:8.1f}  anchor_t_max_full={r['anchor_t_max_full']:8.1f}  coverage={r['coverage_ratio']*100:5.1f}%")
cov_vals = [r["coverage_ratio"] for r in data]
print(f"\ncoverage_ratio: min={min(cov_vals)*100:.1f}% max={max(cov_vals)*100:.1f}% mean={sum(cov_vals)/len(cov_vals)*100:.1f}%")

print()
print("=== 1. per_col_unknown_rate (ON vs anchor_matched, worst col per video/side) ===")
fail_unknown = []
for r in data:
    for side in ("1P", "2P"):
        s = r["sides"][side]
        on_rates = s["per_col_unknown_rate_on"]
        am_rates = s["per_col_unknown_rate_anchor_matched"]
        worst_on_col = max(on_rates, key=lambda c: on_rates[c])
        worst_on = on_rates[worst_on_col]
        worst_am_col = max(am_rates, key=lambda c: am_rates[c])
        worst_am = am_rates[worst_am_col]
        flag_on = "CRITICAL" if worst_on >= PER_COL_UNKNOWN_CRITICAL else ("WARNING" if worst_on >= PER_COL_UNKNOWN_WARNING else "")
        flag_am = "CRITICAL" if worst_am >= PER_COL_UNKNOWN_CRITICAL else ("WARNING" if worst_am >= PER_COL_UNKNOWN_WARNING else "")
        if flag_on or flag_am:
            print(f"{r['video']:6s} {side} ON worst=col{worst_on_col}:{worst_on*100:5.2f}%{flag_on:9s} | anchor_matched worst=col{worst_am_col}:{worst_am*100:5.2f}%{flag_am}")
            if flag_on:
                fail_unknown.append((r["video"], side, "ON", worst_on_col, worst_on))
print(f"ON側 WARNING/CRITICAL発火数: {len(fail_unknown)}")

print()
print("=== 2. per_col_midgame_empty_rate (ON vs anchor_matched, worst col per video/side) ===")
fail_empty = []
for r in data:
    for side in ("1P", "2P"):
        s = r["sides"][side]
        on_rates = s["per_col_midgame_empty_rate_on"]
        am_rates = s["per_col_midgame_empty_rate_anchor_matched"]
        on_n = s["midgame_n_frames_on"]
        am_n = s["midgame_n_frames_anchor_matched"]
        valid_on = {c: v for c, v in on_rates.items() if v == v and on_n >= MIDGAME_COL_MIN_FRAMES}
        valid_am = {c: v for c, v in am_rates.items() if v == v and am_n >= MIDGAME_COL_MIN_FRAMES}
        if not valid_on and not valid_am:
            continue
        worst_on = max(valid_on.values()) if valid_on else float("nan")
        worst_am = max(valid_am.values()) if valid_am else float("nan")
        flag_on = worst_on == worst_on and worst_on >= MIDGAME_COL_EMPTY_CRITICAL
        flag_am = worst_am == worst_am and worst_am >= MIDGAME_COL_EMPTY_CRITICAL
        if flag_on or flag_am:
            print(f"{r['video']:6s} {side} ON worst={worst_on*100:5.1f}% (n={on_n}) | anchor_matched worst={worst_am*100:5.1f}% (n={am_n}) {'ON-CRIT' if flag_on else ''} {'ANCHOR-CRIT' if flag_am else ''}")
            if flag_on:
                fail_empty.append((r["video"], side, worst_on))
print(f"ON側 CRITICAL発火数: {len(fail_empty)}")

print()
print("=== 3. avg_puyo_count (ON vs anchor_matched vs anchor_full, per-side row平均) ===")
ratios = []
for r in data:
    for side in ("1P", "2P"):
        s = r["sides"][side]
        on_v = s["avg_puyo_count_on"]
        am_v = s["avg_puyo_count_anchor_matched"]
        af_v = s["avg_puyo_count_anchor_full"]
        ratio = on_v / am_v if am_v else float("nan")
        ratios.append((r["video"], side, on_v, am_v, af_v, ratio))

print(f"{'video':6s} {'side':3s} {'ON':>8s} {'anc_matched':>12s} {'anc_full':>10s} {'ON/anc_matched':>15s}")
bad = []
for video, side, on_v, am_v, af_v, ratio in sorted(ratios, key=lambda x: x[5]):
    mark = " <0.85!" if ratio < 0.85 else ""
    print(f"{video:6s} {side:3s} {on_v:8.2f} {am_v:12.2f} {af_v:10.2f} {ratio:15.3f}{mark}")
    if ratio < 0.85:
        bad.append((video, side, ratio))
print(f"\nC1 (avg_puyo_count ON/anchor_matched < 0.85) 発火数: {len(bad)}")
for b in bad:
    print(" ", b)

# 全体平均比
tot_on = sum(x[2] for x in ratios)
tot_am = sum(x[3] for x in ratios)
print(f"\n全体合算 avg_puyo_count: ON_sum_of_means={tot_on:.1f} anchor_matched_sum_of_means={tot_am:.1f} ratio={tot_on/tot_am:.4f}")
