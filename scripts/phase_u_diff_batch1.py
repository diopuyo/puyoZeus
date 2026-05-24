"""バッチ 1 の新旧 csv で recognized 列を位置ベース比較。

旧 csv (ユーザレビュー済み、ALL OK or 部分修正) と新 csv の認識結果が
同位置で一致するか確認。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console  # noqa: E402
init_console()

# m1-m13 のユーザレビュー結果 (m14 除く)
# 旧 csv の recognized 列をそのまま truth とするシート (ALL OK)
ALL_OK_SHEETS = ["m1", "m2", "m3", "m6", "m10", "m13"]

# 部分修正シート (旧 csv 1-indexed の id → 真値ラベル文字)
PARTIAL_LABELS: dict[str, dict[int, str]] = {
    "m8": {1: "EM", 2: "PUR", 3: "BLUE", 4: "PUR", 5: "GRN"},
    "m9": {46: "YEL", 47: "RED", 48: "BLUE", 49: "RED", 50: "RED"},
}

# m11 全 50 件 (YPRYE / YYYEE / EBBRR / RPBBP / YBRPP / PPYBB / BYYBR /
#               RRYBY / PYYYP / RYPRR)
M11_LABELS = (
    "Y", "P", "R", "Y", "E", "Y", "Y", "Y", "E", "E",
    "E", "B", "B", "R", "R", "R", "P", "B", "B", "P",
    "Y", "B", "R", "P", "P", "P", "P", "Y", "B", "B",
    "B", "Y", "Y", "B", "R", "R", "R", "Y", "B", "Y",
    "P", "Y", "Y", "Y", "P", "R", "Y", "P", "R", "R",
)
ASCII_TO_REC = {
    "R": "RED", "B": "BLUE", "G": "GRN", "Y": "YEL",
    "P": "PUR", "O": "OJM", "E": "EM",
}


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_user_truth(sheet: str, idx: int, old_rec: str) -> str:
    """idx (1-indexed) のユーザ真値を返す。ALL OK なら old_rec をそのまま。"""
    if sheet in ALL_OK_SHEETS:
        return old_rec
    if sheet == "m11":
        return ASCII_TO_REC[M11_LABELS[idx - 1]]
    if sheet in PARTIAL_LABELS and idx in PARTIAL_LABELS[sheet]:
        return PARTIAL_LABELS[sheet][idx]
    # 部分修正シートで指定外 id は old_rec のまま
    return old_rec


def main() -> int:
    base_old = Path("data/verify/phase_u_batch1_old")
    base_new = Path("data/verify/phase_u_batch1")
    sheets = ["m1", "m2", "m3", "m6", "m8", "m9", "m10", "m11", "m13"]

    total_t = 0
    total_c = 0
    all_mistakes: list = []
    for sh in sheets:
        old_rows = load_csv(base_old / sh / "labels.csv")
        new_rows = load_csv(base_new / sh / "labels.csv")
        if not old_rows or not new_rows:
            print(f"{sh}: missing csv")
            continue
        # 位置ベース対応: (time, side, row, col) → row data
        new_index: dict[tuple, dict] = {}
        for r in new_rows:
            key = (r["time"], r["side"], r["row"], r["col"])
            new_index[key] = r
        correct = 0
        mistakes: list = []
        for old in old_rows:
            idx = int(old["id"])
            old_rec = old["recognized"]
            truth = get_user_truth(sh, idx, old_rec)
            key = (old["time"], old["side"], old["row"], old["col"])
            new = new_index.get(key)
            new_rec = new["recognized"] if new else "EM"  # 新 csv で EMPTY 化
            if new_rec == truth:
                correct += 1
            else:
                mistakes.append((
                    idx, old["side"], f"r{old['row']}c{old['col']}",
                    new_rec, truth,
                ))
        total_t += len(old_rows)
        total_c += correct
        print(f"{sh}: {correct}/{len(old_rows)} ({len(mistakes)} mistakes)")
        for m in mistakes[:5]:
            all_mistakes.append((sh, *m))

    print(f"\nBATCH1 TOTAL (new logic): {total_c}/{total_t} "
          f"({total_c/max(1,total_t)*100:.1f}%)")
    print()
    print("=== Sample mistakes ===")
    for sh, idx, side, pos, new_rec, truth in all_mistakes[:30]:
        print(f"  {sh} id{idx:2d} {side} {pos}: {new_rec} -> {truth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
