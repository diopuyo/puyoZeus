"""ユーザーの char-coded review 結果を violations.csv に反映。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_apply_weak_review \
        --csv data/verify/phase_z_review/weak_video_extra/v04_m03_1137_1167/violations_review/violations.csv \
        --review "GREYBPEEEEE..."

review 文字列は char-coded (E/R/B/G/Y/P/O/?)、シート上の cell 順序と一致。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

CHAR_TO_LABEL = {
    "E": "EM", "R": "RED", "B": "BLUE", "G": "GRN",
    "Y": "YEL", "P": "PUR", "O": "OJM", "?": "??",
    "X": "EX",  # Excluded (effect/movement)
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--review", type=str, required=True,
        help="char-coded review (E/R/B/G/Y/P/O/?/X)、シート順",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: {args.csv} not found")
        return 1

    review = args.review.upper().replace(" ", "").replace("\n", "")
    rows: list[dict] = []
    with args.csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)

    if len(review) != len(rows):
        print(
            f"WARN: review chars {len(review)} != rows {len(rows)}, "
            f"先頭から min で適用"
        )
    n_apply = min(len(review), len(rows))
    n_match = 0
    n_mismatch = 0
    n_excluded = 0
    n_unknown = 0
    for i, r in enumerate(rows[:n_apply]):
        ch = review[i]
        label = CHAR_TO_LABEL.get(ch)
        if label is None:
            print(f"WARN: 不正 char '{ch}' at {i+1}")
            r["your_answer"] = ""
            continue
        if label == "EX":
            r["your_answer"] = "EX"
            n_excluded += 1
            continue
        if label == "??":
            r["your_answer"] = "??"
            n_unknown += 1
            continue
        r["your_answer"] = label
        if r["recognized"] == label:
            n_match += 1
        else:
            n_mismatch += 1

    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_eval = n_match + n_mismatch
    print(f"[apply] {n_apply}/{len(rows)} cells")
    print(f"  evaluated: {n_eval}")
    print(f"    match: {n_match}")
    print(f"    mismatch (真の誤り): {n_mismatch}")
    print(f"  excluded (effect/move): {n_excluded}")
    print(f"  unknown (?): {n_unknown}")
    print(f"\nsaved: {to_windows_path(args.csv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
