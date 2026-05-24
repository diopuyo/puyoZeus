"""残レビュー進捗カウント (一時スクリプト、削除可)."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

SEGMENTS = [
    "v04_m03_1137_1167",
    "v04_m04_1208_1238",
    "v04_m05_1259_1289",
    "v06_m03_385_415",
    "v06_m04_445_475",
    "v06_m05_484_514",
    "v12_m03_387_417",
    "v12_m04_465_495",
    "v12_m05_538_568",
    "v16_m03_323_353",
    "v16_m04_361_391",
    "v16_m05_399_429",
    "v19_m04_445_475",
]
ROOT = Path("data/verify/phase_z_review/weak_video_extra")


def main() -> None:
    total_cells = 0
    total_filled = 0
    for seg in SEGMENTS:
        path = ROOT / seg / "violations_review" / "violations.csv"
        if not path.exists():
            print(f"{seg:<22}  MISSING")
            continue
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        vals = [(r.get("your_answer") or "").strip() for r in rows]
        n = len(vals)
        filled = sum(1 for v in vals if v)
        total_cells += n
        total_filled += filled
        counter = Counter(v if v else "(empty)" for v in vals)
        summary = " ".join(f"{k}={v}" for k, v in sorted(counter.items()))
        pct = 100.0 * filled / n if n else 0.0
        print(f"{seg:<22}  filled: {filled:4d} / {n:4d}  ({pct:5.1f}%)  | {summary}")
    pct = 100.0 * total_filled / total_cells if total_cells else 0.0
    print(f"{'TOTAL':<22}  filled: {total_filled:4d} / {total_cells:4d}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
