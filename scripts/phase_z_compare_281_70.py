"""281.70 の手動 GT vs labels.csv 比較 + suspicious 網羅性測定。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

GT_1P = [
    "EEEEEE", "EEEEEE", "EEEEEE", "EEEEEE",
    "EEEERY", "GEEEPP", "YEERPY", "GGGPGY",
    "YYYGPY", "PGRYPR", "PPGRPR", "GGRRYR",
]
GT_2P = [
    "EEEEEE", "EEEEEE", "EEEEEO", "EEEEEG",
    "OREGEG", "RYEYGP", "ROOOPG", "RPPPGG",
    "YYYGYY", "RGPYRY", "RRGPPR", "GGPYRR",
]

CHAR_TO_LABEL = {
    "E": "EM", "R": "RED", "B": "BLUE", "G": "GRN",
    "Y": "YEL", "P": "PUR", "O": "OJM", "?": "??",
}


def load_labels(csv_path: Path, t_target: str = "281.70") -> dict:
    out = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["time"] != t_target:
                continue
            key = (r["side"], int(r["row"]), int(r["col"]))
            out[key] = (r["recognized"], r.get("suspicious_reasons", ""))
    return out


def main() -> int:
    labels_path = (
        _ROOT / "data/verify/phase_z_review/v18_m03_30_60/labels.csv"
    )
    recognized = load_labels(labels_path)
    if not recognized:
        print("ERROR: t=281.70 not found")
        return 1

    total = 0
    correct = 0
    mismatches: list = []
    for side, gt_grid in (("1P", GT_1P), ("2P", GT_2P)):
        for vrow, line in enumerate(gt_grid):
            for col, ch in enumerate(line):
                gt_label = CHAR_TO_LABEL[ch]
                rec_label, sus_reasons = recognized.get(
                    (side, vrow, col), ("??", ""),
                )
                total += 1
                if rec_label == gt_label:
                    correct += 1
                else:
                    mismatches.append(
                        (side, vrow, col, rec_label, gt_label, sus_reasons),
                    )

    print(f"=== t=281.70 ===")
    print(f"total: {total}, correct: {correct}, "
          f"accuracy: {100.0 * correct / total:.2f}%")
    print(f"mismatches: {len(mismatches)}")

    n_caught = sum(1 for m in mismatches if m[5])
    print(f"suspicious カバー: {n_caught}/{len(mismatches)} "
          f"({100.0 * n_caught / max(1, len(mismatches)):.1f}%)")
    print()
    print("side row col | recognized -> gt | reasons")
    print("-" * 80)
    for side, vrow, col, rec, gt, reasons in mismatches:
        flag = "✓" if reasons else "×"
        print(f"{flag} {side}  r{vrow:2d} c{col} | {rec:5s} -> {gt:5s} | {reasons}")

    # suspicious としてフラグされたが実は正解だった (false positive) 数
    n_false_sus = 0
    n_total_sus = 0
    for (side, vrow, col), (rec, reasons) in recognized.items():
        if reasons:
            n_total_sus += 1
            gt_grid = GT_1P if side == "1P" else GT_2P
            ch = gt_grid[vrow][col]
            gt_label = CHAR_TO_LABEL[ch]
            if rec == gt_label:
                n_false_sus += 1
    print()
    print(f"suspicious 総数: {n_total_sus}, "
          f"うち実は正解 (false positive): {n_false_sus}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
