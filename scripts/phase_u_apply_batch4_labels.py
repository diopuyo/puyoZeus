"""バッチ 4 のユーザレビュー結果を csv に反映する。"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console  # noqa: E402
init_console()

ASCII_TO_REC = {
    "R": "RED", "B": "BLUE", "G": "GRN", "Y": "YEL",
    "P": "PUR", "O": "OJM", "E": "EM",
}

# ユーザのレビュー結果 (ALL OK のシートは省略 = recognized そのまま)
LABELS = {
    "m33": (
        "EEEEE" "PERRP" "YGRRP" "PYGGY" "YRGRG"
        "RYGYG" "RYYPP" "PPRYE" "PRRPP" "GOOPG"
    ),
    "m40": (
        "PPEEG" "GEOYP" "GBGBE" "EPBEB" "PEEEE"
        "EEEEE" "EEEEE" "EEEEE" "EEEEE" "EEEEE"
    ),
    "m42": (
        "EOEEE" "ROYEP" "BBYYE" "EEEEE" "EPYYP"
        "YBRRB" "RRPRR" "PPYRY" "PBOYO" "BOBRP"
    ),
    "m44": (
        "YGRBG" "BRGOG" "RBGYR" "ORRYR" "RYRRR"
        "ROGRG" "ORYYR" "RRYYB" "BERYR" "YRRRR"
    ),
}


def update_csv(csv_path: Path, labels_str: str) -> int:
    """csv の your_answer 列をラベルで上書き。"""
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    chars = [c for c in labels_str.upper() if c in ASCII_TO_REC]
    if len(chars) != len(rows):
        print(f"  WARN: csv {len(rows)} vs labels {len(chars)}")
        return 0
    n_diff = 0
    for row, ch in zip(rows, chars):
        new_truth = ASCII_TO_REC[ch]
        if row["your_answer"] != new_truth:
            n_diff += 1
        row["your_answer"] = new_truth
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return n_diff


def main() -> int:
    base = Path("data/verify/phase_u_batch4")
    total = 0
    for sheet, labels_str in LABELS.items():
        csv_path = base / sheet / "labels.csv"
        if not csv_path.exists():
            print(f"{sheet}: csv missing")
            continue
        diff = update_csv(csv_path, labels_str)
        total += diff
        print(f"{sheet}: {diff} cells corrected")
    print(f"\nTOTAL: {total} corrections")
    return 0


if __name__ == "__main__":
    sys.exit(main())
