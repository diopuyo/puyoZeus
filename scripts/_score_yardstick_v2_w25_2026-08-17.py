"""W25根治 案4 (2026-08-17) の統一測定 構成F vs 構成F+本フラグ 採点比較。

`scripts/_score_yardstick_v2_r2w10_2026-08-17.py` の採点ロジック
(score_one/_load_scores) をそのまま再利用し、以下2構成を55有効盤面で比較する:

    F  : data/indicators_v2/yardstick_v2_boards_f_2026-08-17/ (既存, 再収集不要)
    w25: data/indicators_v2/yardstick_v2_boards_w25_2026-08-17/
         (= 構成F + --enable-ojama-cnn-override-warmup)

W16教訓に基づき、2段階の公平化 (stage1: 共通突合部分集合、stage2: 同一
フレーム限定) で比較する (r2w10 スクリプトと同一手法)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._score_yardstick_v2_w25_2026-08-17 --score f
    PYTHONPATH=. ./venv/bin/python -m scripts._score_yardstick_v2_w25_2026-08-17 --score w25
    PYTHONPATH=. ./venv/bin/python -m scripts._score_yardstick_v2_w25_2026-08-17 --compare
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent

# ファイル名にハイフンを含むため import 文でなく importlib で動的 import する。
_sc = importlib.import_module("scripts._score_yardstick_v2_2026-08-14")

YARDSTICK_DIR = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14"
SCORING_DIR = YARDSTICK_DIR / "scoring_ablation"

TAGS: tuple[str, ...] = ("f", "w25")
TAG_LABEL: dict[str, str] = {
    "f": "F: 構成F (本番採用構成+R2+W10ガード+持続誤認26件修正2点+W23根治)",
    "w25": "w25: F + --enable-ojama-cnn-override-warmup (W25根治案4)",
}
NPZ_DIRS: dict[str, Path] = {
    "f": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_f_2026-08-17",
    "w25": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_w25_2026-08-17",
}


def score_one(tag: str) -> None:
    """構成tag のnpzを採点し、score_{tag}.json を書く (score_one と同一形式)。"""
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
    out_path = SCORING_DIR / f"score_w25cmp_{tag}.json"
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
        path = SCORING_DIR / f"score_w25cmp_{tag}.json"
        if not path.exists():
            raise FileNotFoundError(f"score_w25cmp_{tag}.json が無い。先に --score {tag} を実行すること")
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

    print("\n=== (3) F->w25 で is_correct が変化したシート ===")
    n_newly_correct = 0
    n_newly_wrong = 0
    for sid in stage1_ids:
        f_row = by_sheet["f"][sid]
        w_row = by_sheet["w25"][sid]
        f_acc = f_row["n_correct_excl_row0"] == f_row["n_cells_excl_row0"]
        w_acc = w_row["n_correct_excl_row0"] == w_row["n_cells_excl_row0"]
        if not f_acc and w_acc:
            n_newly_correct += 1
            print(f"  {sid}: F=不完全一致 -> w25=完全一致 (改善)")
        elif f_acc and not w_acc:
            n_newly_wrong += 1
            print(f"  {sid}: F=完全一致 -> w25=不完全一致 (新規悪化)")
    print(f"\n[summary] シート単位 改善={n_newly_correct} 新規悪化={n_newly_wrong}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", choices=["f", "w25"])
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
