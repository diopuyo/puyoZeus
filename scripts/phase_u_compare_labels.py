"""ユーザのラベル (ASCII 1 文字) と CSV の認識結果を一括比較する。

入力: ユーザのラベル文字列 (10 行 × 5 文字 = 50 文字) を sheet ごとに辞書で渡す。
出力: 集計と誤認 ID リスト。
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

# ASCII -> recognized 列の文字列マッピング
ASCII_TO_REC: dict[str, str] = {
    "R": "RED", "B": "BLUE", "G": "GRN", "Y": "YEL",
    "P": "PUR", "O": "OJM", "E": "EM",
}


def compare_one(sheet_dir: Path, user_labels: str) -> dict:
    """1 シート分を比較。user_labels は 50 文字 (改行除く) を想定。"""
    csv_path = sheet_dir / "labels.csv"
    if not csv_path.exists():
        return {"error": f"csv not found: {csv_path}"}
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    user_chars = [c for c in user_labels.upper() if c in ASCII_TO_REC]
    if len(user_chars) != len(rows):
        return {
            "error": (
                f"length mismatch: csv={len(rows)} chars={len(user_chars)} "
                f"({user_labels!r})"
            ),
        }
    correct = 0
    mistakes: list[tuple[int, str, str, str, str]] = []
    for i, (row, ch) in enumerate(zip(rows, user_chars), start=1):
        rec = row["recognized"]
        truth = ASCII_TO_REC[ch]
        if rec == truth:
            correct += 1
        else:
            mistakes.append((
                i, row["side"], f"r{row['row']}c{row['col']}",
                rec, truth,
            ))
    return {
        "total": len(rows),
        "correct": correct,
        "mistakes": mistakes,
    }


def main() -> int:
    # batch 2 user labels
    LABELS = {
        "m4": "RRBBP" "PPYRB" "RRBYY" "RYPPY" "YYPYY"
              "BRBBB" "BPPYR" "PYYPP" "YRRRY" "YPYYY",
        "m5": "RPRYY" "BBBRR" "YPYYR" "RYPPR" "YYPYY"
              "RBYYR" "YYBBP" "RYPYY" "RRYPP" "RYYPR",
        "m7": "YYGGY" "YGGRR" "RBRBP" "GYRPP" "PPGYY"
              "YGGPP" "PGGYP" "YPGRG" "GGGYY" "GGYPG",
        "m15": "BGBYY" "YPYYB" "GYPPY" "BGPBB" "PPGPB"
               "GPBYP" "YPGBB" "YYYPG" "GBPPG" "BYBPY",
        "m16": "GGRGY" "BYBGB" "RYBGB" "BYBYR" "GBBRY"
               "YRGGG" "RRBRR" "RGGYG" "YBYBB" "YBBBG",
        "m17": "EYPBR" "RBBRY" "PPBBB" "PYRPP" "RYYYR"
               "RYRYY" "PYYPR" "BBBRP" "YPPPY" "YBBBR",
        "m19": "RYRYR" "BYBYG" "YGRBB" "YRRYB" "GGGBR"
               "RYBYB" "BGYYB" "YYBBB" "YRRBR" "RBYBB",
        "m20": "GGGGY" "BGPPY" "BPGBB" "YGBGB" "PGYBY"
               "GPGPY" "YYGPG" "YGGPB" "GYPYG" "BGGGP",
        "m23": "RYRRR" "BBPPY" "RRRPR" "PPPRR" "YYYRY"
               "YPRBR" "PPPPR" "BBBPP" "BPPPR" "RRPRY",
        "m25": "EBPBB" "GRGGG" "GBBBP" "GBGGG" "PGRBR"
               "GPRBB" "RGPRG" "GGPGG" "BRPGB" "BBBRP",
    }

    base = Path("data/verify/phase_u_batch2")
    total_t = 0
    total_c = 0
    all_mistakes: list[tuple[str, int, str, str, str, str]] = []
    for name in sorted(LABELS.keys()):
        sheet_dir = base / name
        result = compare_one(sheet_dir, LABELS[name])
        if "error" in result:
            print(f"{name}: ERROR {result['error']}")
            continue
        total_t += result["total"]
        total_c += result["correct"]
        n_mis = len(result["mistakes"])
        print(f"{name}: {result['correct']}/{result['total']} ({n_mis} mistakes)")
        for mi in result["mistakes"]:
            all_mistakes.append((name, *mi))
    print()
    print(f"TOTAL: {total_c}/{total_t} ({total_c/max(1,total_t)*100:.1f}%)")
    print()
    print("=== Mistakes (top 60) ===")
    for sheet, i, side, pos, rec, truth in all_mistakes[:60]:
        print(f"  {sheet} id{i:2d} {side} {pos}: {rec} -> {truth}")

    # 誤認パターン集計
    pattern_count: dict[tuple[str, str], int] = {}
    for sheet, i, side, pos, rec, truth in all_mistakes:
        key = (rec, truth)
        pattern_count[key] = pattern_count.get(key, 0) + 1
    print()
    print("=== Mistake patterns ===")
    for (rec, truth), count in sorted(
        pattern_count.items(), key=lambda x: -x[1],
    ):
        print(f"  {rec:5s} -> {truth:5s}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
