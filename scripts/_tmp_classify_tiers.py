# 索引タイトルから各動画のティアを分類し、ティア別在庫+盤面データ有無を集計
import csv
import glob
import os
import re
from collections import Counter

TIERS = [
    ("マスター", ["マスター"]),
    ("チャレンジャー", ["チャレンジャー"]),
    ("S級", ["S級", "Sリーグ", "S1", "S2", "S3", "S4"]),
    ("A級", ["A級", "A・B", "Aリーグ", "A1", "A2"]),
]


def classify(title: str) -> str:
    for name, kws in TIERS:
        for kw in kws:
            if kw in title:
                return name
    return "不明"


# 索引2つを読む (video_id -> title)
id_title: dict[str, str] = {}
for path, idcol in [("data/phase_e_dl_index.tsv", "video_id"),
                    ("data/_dl_expand.tsv", None)]:
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8", errors="replace") as f:
        if path.endswith("phase_e_dl_index.tsv"):
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                vid = f"video_{row['video_idx']}"
                id_title[vid] = row.get("title", "")
        else:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3 and parts[0].startswith("video_c"):
                    id_title[parts[0]] = parts[2]

# 盤面データ有無
lean_ids = {os.path.basename(p)[:-4] for p in glob.glob("data/indicators_v2/boards_lean_fixed/*.npz")}  # c1..
lean_ids = {f"video_{x}" for x in lean_ids}
study_ids = {os.path.basename(p)[:-4] for p in glob.glob("data/indicators_v2/boards/v*.npz")}
study_ids = {x.replace("v", "video_") for x in study_ids}

tier_counts = Counter()
tier_with_board = Counter()
examples: dict[str, list[str]] = {}
for vid, title in id_title.items():
    t = classify(title)
    tier_counts[t] += 1
    has_board = (vid in lean_ids) or (vid in study_ids) or \
                (vid.replace("video_", "video_c") in lean_ids)
    # c系の盤面はboards_lean_fixedにc番号で入る
    ckey = "c" + vid.split("_")[1] if vid.startswith("video_c") else None
    if ckey and f"video_{ckey}" in lean_ids:
        has_board = True
    if has_board:
        tier_with_board[t] += 1
    examples.setdefault(t, []).append(f"{vid}={title[:22]}")

print("=== ティア別 動画数 / うち盤面データあり ===")
for t, _ in TIERS + [("不明", [])]:
    print(f"{t:12s}: {tier_counts.get(t,0):3d}本  (盤面あり {tier_with_board.get(t,0)})")
print(f"\n索引総数: {len(id_title)}")
print("\n=== 各ティアの例 ===")
for t, _ in TIERS + [("不明", [])]:
    for e in examples.get(t, [])[:3]:
        print(f"  [{t}] {e}")
