# 全30窓CSVがXII列を持ち・行数がベースラインと一致するか検証
import csv
from pathlib import Path

XII = ["saturated_chain_count", "ignition_point_count", "multi_color_ignition",
       "sub_chain_count", "simultaneous_pop_richness"]
study = Path("data/indicators_v2/study")
backup = Path("data/indicators_v2/study_backup_2026-07-20")

windows = []
for v in range(29, 39):
    for suf in ["", "_gap", "_mid"]:
        windows.append(f"v{v}{suf}")

bad = []
total_rows = 0
for w in windows:
    p = study / f"{w}.csv"
    if not p.exists():
        bad.append(f"{w}: 欠損")
        continue
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    total_rows += len(rows)
    has_xii = all(c in rows[0] for c in XII) if rows else False
    # 行数をバックアップと比較
    bp = backup / f"{w}.csv"
    brows = len(list(csv.DictReader(open(bp, encoding="utf-8")))) if bp.exists() else -1
    match = "=" if brows == len(rows) else f"≠(旧{brows})"
    flag = "" if (has_xii and brows == len(rows)) else "  ★"
    if not has_xii:
        bad.append(f"{w}: XII列なし")
    print(f"{w:10s} rows={len(rows):5d} 旧{match:12s} XII={'○' if has_xii else '×'}{flag}")

print(f"\n合計 {len(windows)}窓 {total_rows}行")
print("問題:", bad if bad else "なし")
