# 打ち合い計測用の23本を再現可能ランダムで選定(全てboards_lean_fixed+exchange_labels有)
import glob
import os
import random

lean = {os.path.basename(p)[:-4] for p in glob.glob("data/indicators_v2/boards_lean_fixed/*.npz")}


def avail(prefix_range):
    return [f"c{i}" for i in prefix_range if f"c{i}" in lean]


challenger_pool = avail(range(4, 34))   # c4-c33 = 新おいうリーグ チャレンジャー
master_pool = avail(range(34, 82))      # c34-c81 = マスター題
s_rank = avail([82, 83, 84])            # c82-84 = S級決定戦

rng = random.Random(20260722)
challenger = sorted(rng.sample(challenger_pool, min(10, len(challenger_pool))),
                    key=lambda s: int(s[1:]))
master = sorted(rng.sample(master_pool, min(10, len(master_pool))),
                key=lambda s: int(s[1:]))

print(f"チャレンジャー候補 {len(challenger_pool)}本 → 選定 {len(challenger)}: {challenger}")
print(f"マスター候補 {len(master_pool)}本 → 選定 {len(master)}: {master}")
print(f"S級 {len(s_rank)}: {s_rank}")
sel = challenger + master + s_rank
print(f"\n計 {len(sel)}本:")
print(" ".join(f"video_{c}" for c in sel))
# exchange_labels に含まれるか確認
import csv
el = "data/indicators_v2/exchange_labels.csv"
vids_in_el = set()
if os.path.exists(el):
    for row in csv.DictReader(open(el, encoding="utf-8", errors="replace")):
        vids_in_el.add(row.get("video_id", ""))
missing = [c for c in sel if f"video_{c}" not in vids_in_el]
print(f"\nexchange_labels 未収載: {missing if missing else 'なし(全て収載済)'}")
