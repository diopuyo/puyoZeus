"""W15-D: Field review 結果をユーザー char-coded で labels.csv に反映。

field review sheet は 1 sheet に複数 frame × 1P/2P (12行×6列) を含む。
ユーザーは各 frame ごとに 12 行 × 6 列を char で送る。

CSV のセル順:
    for each time t:
        for each side (1P, 2P):
            for vrow 0..11:
                for col 0..5:
                    -> cell (id, time, side, row, col, recognized, your_answer)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

CHAR_TO_LABEL: dict[str, str] = {
    "E": "EM", "R": "RED", "B": "BLUE", "G": "GRN",
    "Y": "YEL", "P": "PUR", "O": "OJM",
}


# v18_m03 field2 (2 frames) 用ラベル
# 構造: { frame_index: { "1P": [12 rows of 6 chars], "2P": [...] } }
# 行が足りない場合 (None や少ない行数) はスキップ
LABELS_V18_M03_FIELD2: dict[int, dict[str, list[str]]] = {
    0: {  # 1 frame目
        "1P": [
            "EEEEEE",
            "EEEEEE",
            "EGPEEE",
            "EGPEEE",
            "EGYEEE",
            "GYYERG",
            "PPPERG",
            "GGRPGY",
            "YYYGPY",
            "PGRYPR",
            "PPGRPE",
            "GGRRYR",
        ],
        "2P": [
            "EEEEEE",
            "EEEEEE",
            "EPEEEG",
            "EGGEEG",
            "PPGEEP",
            "PYYERG",
            "RPPPGG",
            "YYYGYY",
            "RGPYRY",
            "RRGPPR",
            "GGPYRR",
            # 12 行目 (vrow=11) はユーザー未提供 → 空
        ],
    },
    1: {  # 2 frame目
        "1P": [
            "EEEEEE",
            "OEEEEE",
            "REEEEE",
            "POEYER",
            "YGERPR",
            "GYERYP",
            "YYERYP",
            "GGGPPR",
            "YYYGGY",
            "PGRYPR",
            "PPGRPR",
            "GGRRYR",
        ],
        "2P": [],  # ユーザー: なし
    },
}


def apply_to_csv(csv_path: Path, labels: dict[int, dict[str, list[str]]]) -> int:
    if not csv_path.exists():
        print(f"missing csv: {csv_path}")
        return 0

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # 各 frame の time → frame_idx を構築
    seen_times: list[float] = []
    for r in rows:
        t = float(r[1])
        if t not in seen_times:
            seen_times.append(t)
    print(f"detected frames: {len(seen_times)} times {seen_times}")

    # cell ごとに lookup
    n_filled = 0
    n_skipped = 0
    for r in rows:
        try:
            t = float(r[1])
            side = r[2]
            vrow = int(r[3])
            col = int(r[4])
        except ValueError:
            continue
        if t not in seen_times:
            continue
        frame_idx = seen_times.index(t)
        side_labels = labels.get(frame_idx, {}).get(side, [])
        if vrow >= len(side_labels):
            r[6] = ""
            n_skipped += 1
            continue
        line = side_labels[vrow]
        if col >= len(line):
            r[6] = ""
            n_skipped += 1
            continue
        ch = line[col]
        if ch == "?":
            r[6] = ""
            n_skipped += 1
        elif ch in CHAR_TO_LABEL:
            r[6] = CHAR_TO_LABEL[ch]
            n_filled += 1
        else:
            r[6] = ""
            n_skipped += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(
        f"applied: {n_filled} labels, {n_skipped} skipped → "
        f"{to_windows_path(csv_path)}"
    )
    return n_filled


def main() -> int:
    target = Path(
        "data/verify/phase_w_review/v18_m03_field2/labels.csv"
    )
    apply_to_csv(target, LABELS_V18_M03_FIELD2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
