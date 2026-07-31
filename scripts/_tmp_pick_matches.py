# 勝敗strict・適度な長さの試合を動画横断で列挙し、サンプル候補を選ぶ
import json
from pathlib import Path

wd = Path("data/indicators_v2/winners")
cands = []
for vid in range(30, 39):
    p = wd / f"video_{vid}.json"
    if not p.exists():
        continue
    data = json.load(open(p, encoding="utf-8"))
    for g in data["games"]:
        dur = g["end_sec"] - g["start_sec"]
        if g.get("confidence") == "strict" and 55 <= dur <= 130:
            cands.append((f"video_{vid}", g["game_abs_idx"], g["start_sec"],
                          g["end_sec"], dur, g["winner"]))

for c in cands:
    print(f"{c[0]} game{c[1]:2d}  {c[2]:6.0f}-{c[3]:6.0f}s  ({c[4]:3.0f}s)  勝者={c[5]}")
