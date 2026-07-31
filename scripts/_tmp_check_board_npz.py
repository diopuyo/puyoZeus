# 一時検証: study 盤面 npz と study CSV の行対応を確認する
import csv
from pathlib import Path

import numpy as np

for vid in ["v29", "v30"]:
    d = np.load(f"data/indicators_v2/boards/{vid}.npz", allow_pickle=False)
    print(f"=== {vid} ===")
    for k in d.files:
        print(" ", k, d[k].shape, d[k].dtype)
    csv_path = Path(f"data/indicators_v2/study/{vid}.csv")
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    print("  csv rows:", len(rows), " cols:", len(rows[0]))
    # メタが一致するかサンプル確認
    if "t_secs" in d.files:
        t_npz = d["t_secs"]
        t_csv = np.array([float(r["t_sec"]) for r in rows])
        n = min(len(t_npz), len(t_csv))
        print("  t_sec 一致率(先頭n):", float(np.mean(t_npz[:n] == t_csv[:n])), " n_npz:", len(t_npz), " n_csv:", len(t_csv))
