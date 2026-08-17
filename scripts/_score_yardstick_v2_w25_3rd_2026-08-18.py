"""W25根治 第3弾・最終 (2026-08-18) の統一測定 構成F vs 構成F+第3弾フラグ 採点比較。

`scripts/_score_yardstick_v2_2026-08-14.py` の採点ロジックをそのまま再利用し
(ファイル名にハイフンを含むため importlib で動的 import)、以下2構成を55有効
盤面で比較する:

    F     : data/indicators_v2/yardstick_v2_boards_f_2026-08-17/ (既存, 再収集不要)
    w25_3rd: data/indicators_v2/yardstick_v2_boards_w25_3rd_2026-08-18/
             (= 構成F + --enable-ojama-write-accounting-guard)

注記: 正解ラベル (data/verify/yardstick_v2_2026-08-14/labels_final_from_
user.json) は本スクリプト実行時点の working-tree 状態をそのまま使う
(ラベル修正コーダの5セル修正が反映されている場合はそれを使う、
実行ログに mtime を明記する)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._score_yardstick_v2_w25_3rd_2026-08-18 --score f
    PYTHONPATH=. ./venv/bin/python -m scripts._score_yardstick_v2_w25_3rd_2026-08-18 --score w25_3rd
    PYTHONPATH=. ./venv/bin/python -m scripts._score_yardstick_v2_w25_3rd_2026-08-18 --compare
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent

_sc = importlib.import_module("scripts._score_yardstick_v2_2026-08-14")

YARDSTICK_DIR = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14"
SCORING_DIR = YARDSTICK_DIR / "scoring_ablation"
LABELS_PATH = YARDSTICK_DIR / "labels_final_from_user.json"

TAGS: tuple[str, ...] = ("f", "w25_3rd")
TAG_LABEL: dict[str, str] = {
    "f": "F: 構成F (本番採用構成+R2+W10ガード+持続誤認26件修正2点+W23根治)",
    "w25_3rd": "w25_3rd: F + --enable-ojama-write-accounting-guard (W25根治第3弾)",
}
NPZ_DIRS: dict[str, Path] = {
    "f": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_f_2026-08-17",
    "w25_3rd": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_w25_3rd_2026-08-18",
}


def score_one(tag: str) -> None:
    mtime = os.path.getmtime(LABELS_PATH)
    import datetime
    print(f"[labels] {LABELS_PATH} mtime={datetime.datetime.fromtimestamp(mtime)}")
    gts = _sc.load_ground_truth()
    npz_dir = NPZ_DIRS[tag]
    index = _sc.load_npz_index(npz_dir)
    rows = []
    for gt in gts:
        rec, method = _sc.match_record(gt, index)
        if rec is None:
            rows.append({**_sc._row_meta(gt), "match_method": "miss"})
            continue
        mask_ex, match_ex = _sc.cell_mask_and_match(gt["corrected_grid"], rec["grid"], include_row0=False)
        rows.append({
            **_sc._row_meta(gt),
            "match_method": method,
            "n_cells_excl_row0": int(mask_ex.sum()),
            "n_correct_excl_row0": int(match_ex.sum()),
            "npz": rec["npz"], "frame_idx": rec["frame_idx"], "t_sec": rec["t_sec"],
        })
    SCORING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCORING_DIR / f"score_w25_3rd_{tag}.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    n_miss = sum(1 for r in rows if r["match_method"] == "miss")
    n_exact = sum(1 for r in rows if r["match_method"] == "exact")
    tot = sum(r.get("n_cells_excl_row0", 0) for r in rows)
    cor = sum(r.get("n_correct_excl_row0", 0) for r in rows)
    acc = cor / tot if tot else float("nan")
    print(f"[score:{tag}] miss={n_miss} exact={n_exact} cells(行0除)={tot} correct={cor} acc={acc:.4%}")
    print(f"[score:{tag}] -> {out_path}")


def _load_scores() -> dict[str, list[dict[str, Any]]]:
    out = {}
    for tag in TAGS:
        path = SCORING_DIR / f"score_w25_3rd_{tag}.json"
        if not path.exists():
            raise FileNotFoundError(f"score_w25_3rd_{tag}.json が無い。先に --score {tag} を実行すること")
        out[tag] = json.loads(path.read_text(encoding="utf-8"))
    return out


def compare_all() -> None:
    scores = _load_scores()
    by_sheet = {tag: {r["sheet_id"]: r for r in scores[tag]} for tag in TAGS}

    print("\n=== (0) 生の全体一致率 ===")
    for tag in TAGS:
        rows = scores[tag]
        c_ex, t_ex, acc_ex = _sc._accuracy(rows, "n_correct_excl_row0", "n_cells_excl_row0")
        n_miss = sum(1 for r in rows if r["match_method"] == "miss")
        print(f"  [{tag}:{TAG_LABEL[tag]}] miss={n_miss} acc={acc_ex:.4%}({c_ex}/{t_ex})")

    stage1_ids = set(by_sheet["f"])
    for tag in TAGS:
        stage1_ids &= {sid for sid, r in by_sheet[tag].items() if r["match_method"] != "miss"}
    print(f"\n=== (1) stage1: 共通突合部分集合 (n={len(stage1_ids)}/55) ===")
    for tag in TAGS:
        sub = [by_sheet[tag][sid] for sid in stage1_ids]
        c_ex, t_ex, acc_ex = _sc._accuracy(sub, "n_correct_excl_row0", "n_cells_excl_row0")
        print(f"  [{tag}:{TAG_LABEL[tag]}] 行0除={acc_ex:.4%}({c_ex}/{t_ex})")

    stage2_ids = {sid for sid in stage1_ids if all(by_sheet[t][sid]["match_method"] == "exact" for t in TAGS)}
    print(f"\n=== (2) stage2: 同一フレーム限定 (n={len(stage2_ids)}/55) ===")
    for tag in TAGS:
        sub = [by_sheet[tag][sid] for sid in stage2_ids]
        c_ex, t_ex, acc_ex = _sc._accuracy(sub, "n_correct_excl_row0", "n_cells_excl_row0")
        print(f"  [{tag}:{TAG_LABEL[tag]}] 行0除={acc_ex:.4%}({c_ex}/{t_ex})")

    print("\n=== (3) F->w25_3rd で is_correct が変化したシート ===")
    n_newly_correct = n_newly_wrong = 0
    for sid in stage1_ids:
        f_row = by_sheet["f"][sid]
        w_row = by_sheet["w25_3rd"][sid]
        f_acc = f_row["n_correct_excl_row0"] == f_row["n_cells_excl_row0"]
        w_acc = w_row["n_correct_excl_row0"] == w_row["n_cells_excl_row0"]
        if not f_acc and w_acc:
            n_newly_correct += 1
            print(f"  {sid}: F=不完全一致 -> w25_3rd=完全一致 (改善)")
        elif f_acc and not w_acc:
            n_newly_wrong += 1
            print(f"  {sid}: F=完全一致 -> w25_3rd=不完全一致 (新規悪化)")
    print(f"\n[summary] シート単位 改善={n_newly_correct} 新規悪化={n_newly_wrong}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", choices=["f", "w25_3rd"])
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.score:
        score_one(args.score)
        return
    if args.compare:
        compare_all()
        return
    ap.print_help()


if __name__ == "__main__":
    main()
