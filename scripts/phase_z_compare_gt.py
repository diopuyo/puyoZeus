"""手動 GT (複数 frame) と labels.csv の recognized を比較。

Phase Z-1 半自動 GT ツールの精度評価:
    - 各 frame で accuracy 計測
    - suspicious カバレッジ + false positive 比率
    - 総合サマリ (cell 単位平均)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 手動 GT (row=0 が最上段、col=0 が左端、char-coded E/R/B/G/Y/P/O)
GT_FRAMES: dict[str, dict[str, list[str]]] = {
    "281.70": {
        "1P": [
            "EEEEEE", "EEEEEE", "EEEEEE", "EEEEEE",
            "EEEERY", "GEEEPP", "YEERPY", "GGGPGY",
            "YYYGPY", "PGRYPR", "PPGRPR", "GGRRYR",
        ],
        "2P": [
            "EEEEEE", "EEEEEE", "EEEEEO", "EEEEEG",
            "OREGEG", "RYEYGP", "ROOOPG", "RPPPGG",
            "YYYGYY", "RGPYRY", "RRGPPR", "GGPYRR",
        ],
    },
    "290.20": {
        "1P": [
            "EEEEPG", "OEERPG", "REEPGP", "POEPRY",
            "YGERRY", "GYEOPP", "YYORPY", "GGGPGY",
            "YYYGPY", "PGRYPR", "PPGRPR", "GGRRYR",
        ],
        "2P": [
            "YEEEEE", "YEEERP", "PGEERP", "PREEYP",
            "OREEGG", "RYYYPP", "ROOOPG", "RPPPGG",
            "YYYGYY", "RGPYRY", "RRGPPR", "GGPYRR",
        ],
    },
    "305.20": {
        "1P": [
            "EEEEEP", "OPEGEY", "RPEYPG", "POEYPR",
            "YGERPR", "GYERYP", "YYERYP", "GGGPPR",
            "YYYGGY", "PGRYPR", "PPGRPR", "GGRRYR",
        ],
        # 2P は連鎖中のため割愛
    },
}

CHAR_TO_LABEL = {
    "E": "EM", "R": "RED", "B": "BLUE", "G": "GRN",
    "Y": "YEL", "P": "PUR", "O": "OJM", "?": "??",
}


def load_labels_for_time(csv_path: Path, t_target: str) -> dict:
    out = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["time"] != t_target:
                continue
            key = (r["side"], int(r["row"]), int(r["col"]))
            out[key] = (r["recognized"], r.get("suspicious_reasons", ""))
    return out


def evaluate_frame(
    csv_path: Path,
    t_str: str,
    sides_gt: dict[str, list[str]],
) -> dict:
    """1 frame 分の accuracy + suspicious カバー率を計算。"""
    recognized = load_labels_for_time(csv_path, t_str)
    total = 0
    correct = 0
    mismatches: list = []
    for side, gt_grid in sides_gt.items():
        for vrow, line in enumerate(gt_grid):
            for col, ch in enumerate(line):
                gt_label = CHAR_TO_LABEL[ch]
                rec, sus = recognized.get((side, vrow, col), ("??", ""))
                total += 1
                if rec == gt_label:
                    correct += 1
                else:
                    mismatches.append((side, vrow, col, rec, gt_label, sus))
    n_caught = sum(1 for m in mismatches if m[5])
    # 計測対象 cell の suspicious 総数 / うち実は正解
    n_total_sus = 0
    n_false_sus = 0
    for (side, vrow, col), (rec, sus) in recognized.items():
        if side not in sides_gt:
            continue
        if not sus:
            continue
        n_total_sus += 1
        gt_grid = sides_gt[side]
        gt_label = CHAR_TO_LABEL[gt_grid[vrow][col]]
        if rec == gt_label:
            n_false_sus += 1
    return {
        "t": t_str,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "n_mismatch": len(mismatches),
        "n_caught": n_caught,
        "coverage": n_caught / max(1, len(mismatches)),
        "n_total_sus": n_total_sus,
        "n_false_sus": n_false_sus,
        "precision": (
            (n_total_sus - n_false_sus) / n_total_sus if n_total_sus else 0.0
        ),
        "mismatches": mismatches,
    }


def main() -> int:
    labels_path = (
        _ROOT / "data/verify/phase_z_review/v18_m03_30_60/labels.csv"
    )
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found")
        return 1

    summaries = []
    print(f"{'t':<8} {'cells':<6} {'acc':<7} {'mm':<4} "
          f"{'caught':<7} {'cov':<7} {'sus':<5} {'fp':<5} {'prec':<7}")
    print("-" * 70)
    for t_str, sides_gt in GT_FRAMES.items():
        if not sides_gt:
            continue
        s = evaluate_frame(labels_path, t_str, sides_gt)
        summaries.append(s)
        print(
            f"{s['t']:<8} {s['total']:<6} "
            f"{s['accuracy']*100:<6.2f}% "
            f"{s['n_mismatch']:<4} "
            f"{s['n_caught']:<7} "
            f"{s['coverage']*100:<6.1f}% "
            f"{s['n_total_sus']:<5} "
            f"{s['n_false_sus']:<5} "
            f"{s['precision']*100:<6.1f}%"
        )

    # 集計
    tot_cells = sum(s["total"] for s in summaries)
    tot_correct = sum(s["correct"] for s in summaries)
    tot_mm = sum(s["n_mismatch"] for s in summaries)
    tot_caught = sum(s["n_caught"] for s in summaries)
    tot_sus = sum(s["n_total_sus"] for s in summaries)
    tot_fp = sum(s["n_false_sus"] for s in summaries)
    print("-" * 70)
    print(
        f"{'ALL':<8} {tot_cells:<6} "
        f"{100.0 * tot_correct / max(1, tot_cells):<6.2f}% "
        f"{tot_mm:<4} "
        f"{tot_caught:<7} "
        f"{100.0 * tot_caught / max(1, tot_mm):<6.1f}% "
        f"{tot_sus:<5} {tot_fp:<5} "
        f"{100.0 * (tot_sus - tot_fp) / max(1, tot_sus):<6.1f}%"
    )

    # 各 frame の mismatch 詳細
    for s in summaries:
        print()
        print(f"=== t={s['t']} mismatch ({s['n_mismatch']}) ===")
        for side, vrow, col, rec, gt, reasons in s["mismatches"]:
            flag = "✓" if reasons else "×"
            short = reasons[:60] + "..." if len(reasons) > 60 else reasons
            print(f"  {flag} {side} r{vrow:2d} c{col} | "
                  f"{rec:5s} -> {gt:5s} | {short}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
