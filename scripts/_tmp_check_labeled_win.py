# 一時検証: labeled_win.csv の構成 (どのファイル由来か、行数、盤面npzで被覆できる割合)
import csv
from collections import Counter

import numpy as np

rows = list(csv.DictReader(open("data/indicators_v2/study/labeled_win.csv", encoding="utf-8")))
print("labeled_win rows:", len(rows))
print("cols:", list(rows[0].keys())[:12], "... total", len(rows[0]))
print("video_id 分布:", Counter(r["video_id"] for r in rows))
if "source_file" in rows[0]:
    print("source_file 分布:", Counter(r["source_file"] for r in rows))

# 盤面npz側 (video_id, side, t_sec) キーの集合
npz_keys = set()
for vid in ["v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38"]:
    d = np.load(f"data/indicators_v2/boards/{vid}.npz", allow_pickle=False)
    for v, s, t in zip(d["video_id"], d["side"], d["t_sec"]):
        npz_keys.add((str(v), str(s), round(float(t), 2)))
print("npz snapshot 総数:", len(npz_keys))

hit = sum(1 for r in rows if (r["video_id"], r["side"], round(float(r["t_sec"]), 2)) in npz_keys)
print(f"labeled_win のうち npz で盤面が引ける行: {hit}/{len(rows)} = {hit/len(rows):.1%}")
