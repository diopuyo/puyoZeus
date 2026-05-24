"""ユーザの新ラベル 46 件を v3 ラベル TSV に反映する。

入力:
    data/verify/cnn_v3_errors_order.tsv (44 件、現状モデル diagnose 順)
    data/verify/ojama_labels_v3.tsv (元 144 ラベル)
    USER_LABELS (この script 内に embed、46 件)

出力:
    data/verify/ojama_labels_v3.tsv (上書き、元はバックアップ)
    data/verify/v3_label_diff.txt (変更ログ)
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

ORDER_TSV = Path("data/verify/cnn_v3_errors_order.tsv")
LABELS_TSV = Path("data/verify/ojama_labels_v3.tsv")
BACKUP_TSV = Path("data/verify/ojama_labels_v3.bak_pre_user_review.tsv")
DIFF_LOG = Path("data/verify/v3_label_diff.txt")

# ユーザがレビューで提示した 46 件のラベル (順序通り)
USER_LABELS: list[str] = [
    "empty", "empty", "empty", "empty", "empty", "empty", "empty",
    "star", "rock", "star", "rock", "moon", "rock", "large",
    "crown", "star", "rock", "rock", "rock", "rock",
    "empty", "empty", "empty", "empty", "empty", "empty",
    "crown", "rock", "rock", "rock", "rock", "rock",
    "empty", "empty", "empty", "rock", "large", "empty", "empty", "empty",
    "moon", "rock", "small", "small", "small", "small",
]


def main() -> int:
    if not BACKUP_TSV.is_file():
        shutil.copy(LABELS_TSV, BACKUP_TSV)
        print(f"backup: {BACKUP_TSV}")

    # order TSV 読み込み
    orders: list[dict] = []
    with open(ORDER_TSV) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            orders.append(r)
    print(f"order entries: {len(orders)}")
    print(f"user labels: {len(USER_LABELS)}")

    # 既存ラベル読み込み
    label_rows: list[dict] = []
    with open(LABELS_TSV) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            label_rows.append(r)
    label_index = {
        (int(r["frame_idx"]), r["side"], int(r["cell_idx"])): r
        for r in label_rows
    }

    # 対応付け: order[i] ↔ USER_LABELS[i] (上から順)
    n_match = min(len(orders), len(USER_LABELS))
    diffs: list[str] = []
    n_changed = 0
    n_unchanged = 0
    for i in range(n_match):
        o = orders[i]
        new_label = USER_LABELS[i]
        key = (int(o["frame_idx"]), o["side"], int(o["cell_idx"]))
        cur = label_index.get(key)
        if cur is None:
            diffs.append(f"[skip] order_idx={i} key={key} ラベルなし")
            continue
        old = cur["label"]
        if old != new_label:
            cur["label"] = new_label
            n_changed += 1
            diffs.append(
                f"[{i:2d}] F{key[0]} {key[1]} S{key[2]} "
                f"({o['video']} t={o['t_sec']}): "
                f"{old} → {new_label} (CNN pred={o['cnn_pred']})"
            )
        else:
            n_unchanged += 1

    if len(orders) < len(USER_LABELS):
        diffs.append(
            f"[warn] user labels {len(USER_LABELS)} > order {len(orders)}, "
            f"残り {len(USER_LABELS) - len(orders)} 件は対応不可"
        )

    print(f"changed: {n_changed}")
    print(f"unchanged: {n_unchanged}")

    # ラベル TSV 上書き
    with open(LABELS_TSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, delimiter="\t",
            fieldnames=["frame_idx", "side", "cell_idx", "label"],
        )
        w.writeheader()
        for r in label_rows:
            w.writerow({k: r[k] for k in
                        ["frame_idx", "side", "cell_idx", "label"]})

    DIFF_LOG.write_text("\n".join(diffs), encoding="utf-8")
    print(f"diff log: {DIFF_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
