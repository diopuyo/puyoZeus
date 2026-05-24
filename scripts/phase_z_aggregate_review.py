"""ユーザーの hard violations レビュー結果を集計し真の accuracy を計算。

violations.csv の id 順 (シート上の順序) に対するユーザー入力 (5 行 × 12 cells)
と、移動中/エフェクトの除外指示を統合して真の誤り数を出す。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# シート上の 5 行 × 12 cells (順序は violations.csv の id 順と一致)
USER_REVIEW = [
    "GGRRYYPEPGGG",  # row 1: i=1..12
    "PERYYPGYPPRY",  # row 2: i=13..24, 5番目=エフェクト
    "RROPPYEGPOGR",  # row 3: i=25..36, 1番目=移動中, 9-12=エフェクト
    "GEPOPRRPPPRY",  # row 4: i=37..48, 3-6=エフェクト
    "PPYEPR",        # row 5: i=49..54
]

# エフェクト・移動中 cell の除外: シートの 1-indexed 全体 id
EXCLUDED_INDICES: set[int] = {
    17,                       # 行2 5番目 (エフェクト)
    25,                       # 行3 1番目 (移動中)
    33, 34, 35, 36,           # 行3 9-12 (エフェクト)
    39, 40, 41, 42,           # 行4 3-6 (エフェクト)
}

CHAR_TO_LABEL = {
    "E": "EM", "R": "RED", "B": "BLUE", "G": "GRN",
    "Y": "YEL", "P": "PUR", "O": "OJM", "?": "??",
}


def main() -> int:
    csv_path = (
        _ROOT
        / "data/verify/phase_z_review/v18_m03_30_60/violations_review_z3f/violations.csv"
    )
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return 1

    rows: list[dict] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"violations.csv: {len(rows)} rows")
    flat_user = "".join(USER_REVIEW)
    print(f"user input: {len(flat_user)} chars")

    n_match = 0
    n_mismatch = 0
    n_excluded = 0
    detail_mismatch: list[tuple[int, str, str, str]] = []
    for i, r in enumerate(rows):
        sheet_idx = i + 1  # 1-indexed
        if sheet_idx > len(flat_user):
            print(f"WARNING: row {sheet_idx} の user 入力なし")
            continue
        user_char = flat_user[sheet_idx - 1]
        user_label = CHAR_TO_LABEL.get(user_char.upper())
        if user_label is None:
            print(f"WARNING: row {sheet_idx} 不正 char '{user_char}'")
            continue
        rec = r["recognized"]
        if sheet_idx in EXCLUDED_INDICES:
            n_excluded += 1
            continue
        if rec == user_label:
            n_match += 1
        else:
            n_mismatch += 1
            detail_mismatch.append(
                (sheet_idx, r["time"], f"{r['side']}r{r['row']}c{r['col']}",
                 f"{rec}->{user_label}"),
            )

    n_eval = n_match + n_mismatch
    print()
    print("=" * 60)
    print("ユーザーレビュー集計 (hard violations 59 件)")
    print("=" * 60)
    print(f"レビュー除外 (移動中/エフェクト): {n_excluded}")
    print(f"評価対象: {n_eval}")
    print(f"  recognized 一致 (false positive): {n_match}")
    print(f"  真の誤り (mismatch):              {n_mismatch}")
    print()

    # 全 cell に対する accuracy 推定
    # 全 cell = 8784, 連鎖中除外 = 4896, 評価対象 = 3888
    # hard 違反 cell = 59, そのうち 真の誤り = n_mismatch, 一致 = n_match, 除外 = n_excluded
    # 真の accuracy = 1 - (真の誤り / 評価対象)
    eval_total_cells = 3888
    true_acc = 1.0 - n_mismatch / eval_total_cells
    print(f"評価対象 cell 数 (連鎖中除外後): {eval_total_cells}")
    print(f"真の誤り cell 数 (レビュー確定): {n_mismatch}")
    print(f"真の accuracy: {100.0 * true_acc:.3f}%")
    print(f"99.5% 目標との差分: {100.0 * true_acc - 99.5:+.3f}pt")
    print()

    if detail_mismatch:
        print("=" * 60)
        print("真の誤り 詳細 (CellRecoveryRefiner 強化対象)")
        print("=" * 60)
        for idx, t, loc, swap in detail_mismatch:
            # 該当 csv row の reasons を表示
            r = rows[idx - 1]
            print(f"  #{idx} t={t} {loc} | {swap:14s} | {r['reasons']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
