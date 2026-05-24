"""未使用動画 + 試合 2 以降の有無を確認 (一時)."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USED = {1, 4, 6, 7, 9, 13, 15}

for v in range(1, 20):
    if v in USED:
        continue
    for ver in ("v5", "v4"):
        p = ROOT / f"data/verify/match_boundaries_{ver}/video_{v:02d}/matches.tsv"
        if not p.exists():
            continue
        with p.open() as f:
            rows = list(csv.reader(f, delimiter="\t"))
        n = len(rows) - 1  # ヘッダ除く
        durations = []
        for r in rows[1:]:
            try:
                durations.append(float(r[2]) - float(r[1]))
            except (IndexError, ValueError):
                pass
        durations_str = ", ".join(f"{d:.0f}s" for d in durations)
        print(
            f"v{v:02d} ({ver}): {n} matches, "
            f"durations=[{durations_str}]"
        )
        break
    else:
        print(f"v{v:02d}: no boundary")
