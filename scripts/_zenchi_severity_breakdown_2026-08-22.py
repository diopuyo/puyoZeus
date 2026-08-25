import pandas as pd
import re

df = pd.read_csv("data/verify/judgment_scan_zenchi_2026-08-22/suspects.tsv", sep="\t")
blockers = df[df["stage"].isin(["display","both"])].copy()

def extract_disp(ev):
    m = re.search(r"adv_disp=([+-][\d.]+)", ev)
    return float(m.group(1)) if m else None

blockers["adv_disp"] = blockers["evidence"].apply(extract_disp)
blockers["abs_disp"] = blockers["adv_disp"].abs()

print("表示adv絶対値 分布:")
print(blockers["abs_disp"].describe())
print("\n|adv_disp|>=50 (重度, 表示上も強い自信で逆方向) の件数:", (blockers["abs_disp"]>=50).sum(), "/", len(blockers))
print("|adv_disp|>=80 の件数:", (blockers["abs_disp"]>=80).sum())
print("|adv_disp|<10 (軽微,ほぼEVEN境界) の件数:", (blockers["abs_disp"]<10).sum())

# セット別
blockers_set1 = blockers[blockers["t_sec"]<3626.0]
blockers_set2 = blockers[blockers["t_sec"]>=3626.0]
print(f"\nセット1: {len(blockers_set1)}件  セット2: {len(blockers_set2)}件")

# 重度(|adv_disp|>=50)の一覧
severe = blockers[blockers["abs_disp"]>=50].sort_values("t_sec")
print(f"\n=== 重度(|adv_disp|>=50) 一覧 ({len(severe)}件、時刻でグルーピングした先頭のみ表示) ===")
prev_t = -999
for _, row in severe.iterrows():
    if row["t_sec"] - prev_t > 2.0:
        print(f"  t={row['t_sec']:.2f} det={row['detector']} adv_disp={row['adv_disp']:+.1f} : {row['evidence'][:70]}")
    prev_t = row["t_sec"]
