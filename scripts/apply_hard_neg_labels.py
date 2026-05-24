"""
collect_hard_negatives_v02 で抽出した候補（除外指定後）を赤ラベルとして
data/verify/human_labels/ に書き出す。

使い方:
    ./venv/bin/python scripts/apply_hard_neg_labels.py --exclude 28
    （複数除外）./venv/bin/python scripts/apply_hard_neg_labels.py --exclude 5 12 28
"""
from __future__ import annotations
import argparse, csv, datetime as _dt, json, os, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2

from src.board import COLOR_RED

INDIV = Path("data/verify/hard_neg_v02/individual")
TSV = Path("data/verify/hard_neg_v02/report.tsv")
OUT_ROOT = Path("data/verify/human_labels")

CLASS_TO_STR = {1: "red", 2: "blue", 3: "green", 4: "yellow", 5: "purple", 9: "ojama", 0: "empty"}
JP_TO_CODE = {"赤": 1, "青": 2, "緑": 3, "黄": 4, "紫": 5, "お": 9, "空": 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", type=int, nargs="*", default=[],
                        help="除外する idx（複数指定可）")
    args = parser.parse_args()
    excluded = set(args.exclude)

    if not TSV.exists():
        print(f"TSV なし: {TSV}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    with TSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"hard_neg_v02_{ts}"
    patches_dir = out_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_ROOT / f"hard_neg_v02_{ts}.jsonl"

    files_by_idx: dict[int, Path] = {}
    for p in INDIV.glob("*.png"):
        try:
            idx = int(p.name.split("_")[0])
            files_by_idx[idx] = p
        except ValueError:
            continue

    n_kept = 0
    n_skip = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        meta = {
            "kind": "meta",
            "import_ts": ts,
            "tool": "apply_hard_neg_labels.py",
            "source": "hard_neg_v02",
            "policy": "video_02 50 試合中央フレームから赤偽陰性候補を抽出、人手で除外指定後に赤ラベル化",
            "excluded_indices": sorted(excluded),
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        for row in rows:
            idx = int(row["idx"])
            if idx in excluded:
                n_skip += 1
                continue
            src_img = files_by_idx.get(idx)
            if src_img is None:
                continue

            # 4倍拡大されたパッチを 44×44 にリサイズして保存
            img = cv2.imread(str(src_img))
            if img is None:
                continue
            patch_44 = cv2.resize(img, (44, 44), interpolation=cv2.INTER_AREA)
            patch_rel = f"patches/idx{idx:03d}_m{row['match']}_{row['side']}_r{row['row']}_c{row['col']}.png"
            patch_abs = (out_dir / patch_rel).resolve()
            cv2.imwrite(str(patch_abs), patch_44)

            entry = {
                "kind": "correction",
                "frame": f"video_02_match{row['match']}_mid",
                "side": row["side"],
                "row": int(row["row"]),
                "col": int(row["col"]),
                "cnn_predicted": CLASS_TO_STR.get(JP_TO_CODE.get(row["cnn_pred"], 0), "unknown"),
                "true_label": "red",
                "patch_file": str(patch_abs),
                "auto_labeled": True,
                "red_ratio": float(row["red_ratio"]),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n_kept += 1

    print(f"赤ラベル jsonl 出力: {jsonl_path}")
    print(f"  採用: {n_kept}  除外: {n_skip}")
    print(f"  パッチ: {patches_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
