# 一時検証: study CSV 各ファイルの t_sec 範囲 (窓の対応を確定する)
import csv

for vid in ["v29", "v33", "v38"]:
    for suffix in ["", "_mid", "_gap"]:
        path = f"data/indicators_v2/study/{vid}{suffix}.csv"
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        ts = [float(r["t_sec"]) for r in rows]
        print(f"{vid}{suffix}: rows={len(rows)} t_sec=[{min(ts):.1f}, {max(ts):.1f}]")
