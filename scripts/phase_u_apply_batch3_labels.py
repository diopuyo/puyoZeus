"""バッチ 3 のユーザレビュー結果を csv に反映する。

バッチ 4 用スクリプトとの差分:
- '.' を「recognized 列をそのまま採用」のマスク文字として許容
- ALL OK の試合は '.' を 50 文字並べる
- 部分指定 (例: 上 7 行 OK + 下 3 行のみ指定) も '.' で表現可能

シートのセル並び (前提): 5 列 × 10 行、行優先 (上→下、左→右)。
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

ASCII_TO_REC = {
    "R": "RED", "B": "BLUE", "G": "GRN", "Y": "YEL",
    "P": "PUR", "O": "OJM", "E": "EM",
}

# ユーザのレビュー結果。'.' は「recognized そのまま」を意味する。
# 各試合は 5 文字 × 10 行 = 50 文字。
LABELS = {
    # m18: 下から 3 番目 (行 7)、一番右 (列 4) のみ E。それ以外 ALL OK。
    "m18": (
        "....." "....." "....." "....." "....."
        "....." "....." "....E" "....." "....."
    ),
    # m21: ALL OK
    "m21": "." * 50,
    # m22: フル指定。最終行末尾の P は移動中ぷよ + エフェクトあり
    "m22": (
        "RPEEP" "BBBGR" "GRBGG" "RGGBR" "RGPPR"
        "RPBRR" "PPRGR" "GPBGG" "RGGRR" "GBBBP"
    ),
    # m24: フル指定
    "m24": (
        "EGGBG" "BYYEG" "YGGGB" "YRGBY" "BBYGR"
        "YYYRB" "BGGGB" "GGGBY" "YGYGG" "YGBYR"
    ),
    # m26: ALL OK
    "m26": "." * 50,
    # m27: フル指定。先頭行は勝敗決まった後の「やった」ロゴで EMPTY
    "m27": (
        "EEEEE" "EYYYB" "RRGYB" "YEEEE" "EEEEE"
        "EEEEE" "EEEEE" "EEEEE" "EEEEE" "EEEEE"
    ),
    # m28: 上 7 行 OK、下 3 行のみ指定。
    # 行 7 末尾の B、行 8 先頭の B は消滅アニメ中
    "m28": (
        "....." "....." "....." "....." "....."
        "....." "....."
        "GGYPB" "BPPEE" "PGGBG"
    ),
    # m29: フル指定
    "m29": (
        "GRRRB" "PBBPG" "RRRPR" "RGRRG" "RRREE"
        "EEEER" "RRRBR" "RRRBP" "YBBPB" "PPYPY"
    ),
    # m30: フル指定
    "m30": (
        "RYBRR" "EYRYB" "YRBEB" "YPPPR" "RYBRP"
        "PBRYB" "YRBRY" "RPRYY" "ROOOR" "ROOOP"
    ),
    # m32: 上 3 行のみ指定、以下 ALL OK
    "m32": (
        "EPPPP" "RPRRG" "BRRRR"
        "....." "....." "....." "....."
        "....." "....." "....."
    ),
}


def update_csv(csv_path: Path, labels_str: str) -> tuple[int, int]:
    """csv の your_answer 列をマスクで上書き。

    '.' は recognized 列をそのまま採用 (= 認識正解扱い)。
    それ以外の文字は ASCII_TO_REC でマップして上書き。

    Returns:
        (修正セル数, ALL OK 採用セル数)
    """
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    chars = [c for c in labels_str.upper() if c in ASCII_TO_REC or c == "."]
    if len(chars) != len(rows):
        print(f"  WARN: csv {len(rows)} vs labels {len(chars)}")
        return 0, 0
    n_diff = 0
    n_allok = 0
    for row, ch in zip(rows, chars):
        if ch == ".":
            new_truth = row["recognized"]
            n_allok += 1
        else:
            new_truth = ASCII_TO_REC[ch]
        if row["your_answer"] != new_truth:
            n_diff += 1
        row["your_answer"] = new_truth
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return n_diff, n_allok


def main() -> int:
    base = Path("data/verify/phase_u_batch3")
    total_diff = 0
    total_allok = 0
    for sheet, labels_str in LABELS.items():
        csv_path = base / sheet / "labels.csv"
        if not csv_path.exists():
            print(f"{sheet}: csv missing")
            continue
        diff, allok = update_csv(csv_path, labels_str)
        total_diff += diff
        total_allok += allok
        print(f"{sheet}: {diff} corrected, {allok} ALL-OK adopted")
    print(f"\nTOTAL: {total_diff} corrections, {total_allok} ALL-OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
