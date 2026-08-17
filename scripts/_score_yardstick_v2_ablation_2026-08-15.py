"""構成c (OJAMA_FALL系3フラグ同時ON) 単体切り分けアブレーション採点 (2026-08-15)。

`scripts/_score_yardstick_v2_2026-08-14.py` の採点ロジック (突合/セル一致判定/
誤り分類) をそのまま再利用し (ファイル名にハイフンを含むため importlib で動的
import)、以下5構成を55有効盤面で比較する:

    a  : 本番採用構成 (既存npz、再収集不要)
    c1 : a + --enable-ojama-fall-placement-override のみ
    c2 : a + --enable-ojama-fall-entry-hardening のみ
    c3 : a + --enable-ojama-fall-scoped-exit のみ
    c  : a + 3フラグ同時ON (既存npz、再収集不要)

出力は data/verify/yardstick_v2_2026-08-14/scoring_ablation/ に書く
(既存 scoring/ ディレクトリは変更しない)。

使い方:
    python scripts/_score_yardstick_v2_ablation_2026-08-15.py --score c1
    python scripts/_score_yardstick_v2_ablation_2026-08-15.py --score c2
    python scripts/_score_yardstick_v2_ablation_2026-08-15.py --score c3
    python scripts/_score_yardstick_v2_ablation_2026-08-15.py --compare
"""
from __future__ import annotations

import argparse
import importlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent

# ファイル名にハイフンを含むため import 文でなく importlib で動的 import する。
_sc = importlib.import_module("scripts._score_yardstick_v2_2026-08-14")

YARDSTICK_DIR = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14"
SCORING_DIR = YARDSTICK_DIR / "scoring_ablation"

# 5列比較の全タグ (baseline a / 単体3種 / 3枚同時c)。
BASELINE_TAGS: tuple[str, ...] = ("a", "c1", "c2", "c3", "c")

NPZ_DIRS: dict[str, Path] = {
    "a": _sc.NPZ_DIR_A,
    "c": _sc.NPZ_DIR_C,
    "c1": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_c1_placement_2026-08-15",
    "c2": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_c2_entryhard_2026-08-15",
    "c3": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_c3_scopedexit_2026-08-15",
    "c12": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_c12_2026-08-15",
    "c13": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_c13_2026-08-15",
    "c23": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_c23_2026-08-15",
    # c1' (2026-08-15修正版): _confirm_placement_evidence のヒステリシス+
    # chain除外修正を適用した placement_override 単体の再収集。
    "c1p": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_c1p_fix_2026-08-15",
    # W13根治 (2026-08-16): c1p (現行本番採用構成) + --enable-highlight-override
    # 単体の再収集。 c1p が現行の正しい比較基準 (94.08%→95.63%採用済み)。
    "w13": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_w13_2026-08-16",
}


def score_one(tag: str, gts: list[dict[str, Any]]) -> None:
    """構成tagのnpzを採点し、score_{tag}.json を scoring_ablation/ に書く。"""
    npz_dir = NPZ_DIRS[tag]
    index = _sc.load_npz_index(npz_dir)
    rows = []
    for gt in gts:
        rec, method = _sc.match_record(gt, index)
        if rec is None:
            rows.append({**_sc._row_meta(gt), "match_method": "miss"})
            continue
        mask_ex, match_ex = _sc.cell_mask_and_match(gt["corrected_grid"], rec["grid"], include_row0=False)
        mask_in, match_in = _sc.cell_mask_and_match(gt["corrected_grid"], rec["grid"], include_row0=True)
        cells = []
        for r in range(1, _sc.N_ROWS):
            for c_idx in range(_sc.N_COLS):
                if not mask_ex[r, c_idx]:
                    continue
                init_val = int(gt["init_grid"][r][c_idx])
                corr_val = int(gt["corrected_grid"][r][c_idx])
                pred_val = int(rec["grid"][r, c_idx])
                cells.append({
                    "r": r, "c": c_idx, "init": init_val, "correct": corr_val, "pred": pred_val,
                    "is_correct": bool(match_ex[r, c_idx]),
                    "a_error_category": _sc.classify_error(init_val, corr_val),
                })
        rows.append({
            **_sc._row_meta(gt),
            "match_method": method,
            "n_cells_excl_row0": int(mask_ex.sum()),
            "n_correct_excl_row0": int(match_ex.sum()),
            "n_cells_incl_row0": int(mask_in.sum()),
            "n_correct_incl_row0": int(match_in.sum()),
            "cells": cells,
        })
    SCORING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCORING_DIR / f"score_{tag}.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    n_miss = sum(1 for r in rows if r["match_method"] == "miss")
    n_nearest = sum(1 for r in rows if r["match_method"] == "nearest")
    tot = sum(r.get("n_cells_excl_row0", 0) for r in rows)
    cor = sum(r.get("n_correct_excl_row0", 0) for r in rows)
    acc = cor / tot if tot else float("nan")
    print(f"[score:{tag}] miss={n_miss} nearest_fallback={n_nearest} "
          f"cells(行0除)={tot} correct={cor} acc={acc:.4%}")
    print(f"[score:{tag}] -> {out_path}")


def _load_scores(tags: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for tag in tags:
        path = SCORING_DIR / f"score_{tag}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"score_{tag}.json が無い。先に --score {tag} を実行すること (fail-silent禁止)")
        out[tag] = json.loads(path.read_text(encoding="utf-8"))
    return out


def compare_all(tags: tuple[str, ...] = BASELINE_TAGS) -> None:
    """5構成 (a/c1/c2/c3/c) の比較表・分類表をまとめて出力する。"""
    scores = _load_scores(tags)
    SCORING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== (1) 全体一致率 + 突合ズレ (tags={tags}) ===")
    overall_lines = ["config,n_miss,n_nearest,n_mismatch_total,cells_excl_row0,"
                      "correct_excl_row0,acc_excl_row0"]
    for tag in tags:
        rows = scores[tag]
        n_miss = sum(1 for r in rows if r["match_method"] == "miss")
        n_nearest = sum(1 for r in rows if r["match_method"] == "nearest")
        c_ex, t_ex, acc_ex = _sc._accuracy(rows, "n_correct_excl_row0", "n_cells_excl_row0")
        print(f"  [{tag}] miss={n_miss} nearest={n_nearest} "
              f"突合ズレ計={n_miss + n_nearest}/{len(rows)} 行0除一致率={acc_ex:.4%}({c_ex}/{t_ex})")
        overall_lines.append(f"{tag},{n_miss},{n_nearest},{n_miss + n_nearest},{t_ex},{c_ex},{acc_ex:.6f}")
    (SCORING_DIR / "overall_accuracy.csv").write_text("\n".join(overall_lines) + "\n", encoding="utf-8")

    common_ids = set.intersection(*(
        {r["sheet_id"] for r in scores[tag] if r["match_method"] != "miss"} for tag in tags
    ))
    print(f"\n=== (1b) 全構成共通突合部分集合での一致率 (公平比較、n={len(common_ids)}/55) ===")
    common_lines = ["config,cells_excl_row0,correct_excl_row0,acc_excl_row0"]
    for tag in tags:
        sub = [r for r in scores[tag] if r["sheet_id"] in common_ids]
        c_ex, t_ex, acc_ex = _sc._accuracy(sub, "n_correct_excl_row0", "n_cells_excl_row0")
        print(f"  [{tag}] 行0除={acc_ex:.4%}({c_ex}/{t_ex})")
        common_lines.append(f"{tag},{t_ex},{c_ex},{acc_ex:.6f}")
    (SCORING_DIR / "common_subset_accuracy.csv").write_text("\n".join(common_lines) + "\n", encoding="utf-8")

    print("\n=== (2) 層別一致率 (行0除く; 満杯帯/位相/動画別) ===")
    strat_lines = ["axis,stratum,config,cells,correct,accuracy"]
    for axis_name, axis_fn in (
        ("phase", lambda r: r["phase"]),
        ("has_ojama", lambda r: str(r["has_ojama"])),
        ("fill_band", lambda r: _sc._fill_band(r["fill_ratio"])),
        ("video_id", lambda r: r["video_id"]),
    ):
        strata = sorted({axis_fn(r) for r in scores["a"]})
        for st in strata:
            accs = []
            for tag in tags:
                sub = [r for r in scores[tag] if axis_fn(r) == st]
                c_ex, t_ex, acc_ex = _sc._accuracy(sub, "n_correct_excl_row0", "n_cells_excl_row0")
                strat_lines.append(f"{axis_name},{st},{tag},{t_ex},{c_ex},{acc_ex:.6f}")
                accs.append(f"{tag}={acc_ex:.3%}")
            print(f"  [{axis_name}={st}] " + " / ".join(accs))
    (SCORING_DIR / "stratified_accuracy.csv").write_text("\n".join(strat_lines) + "\n", encoding="utf-8")

    print("\n=== (3) 誤り分類表 (aのinit_grid基準、各構成で解消/新規悪化したか) ===")
    # 案3修正 (2026-08-15、過少報告是正): 従来は tags 全構成の N-way
    # intersection (`common_sheets`) を分母にしていたため、 tags に含めた
    # *無関係な他構成* (例: c12/c) の収集欠落シートまで巻き込んで、 評価対象の
    # 構成 (例: c1) 自身は突合できているシートの regression まで丸ごと
    # 欠落扱いになっていた (実測: c1 単体の新規劣化 46 セルが 20 セルに
    # 過少報告される事故、 data/verify/yardstick_v2_2026-08-14/
    # scoring_ablation/_diag_c1_new_regressions_2026-08-15.json で確定)。
    # 各構成 t の集計は「a と t のペアワイズ共通シート」のみを分母にし、
    # 他の tags の欠落から独立させる。 n_cells_gt / wrong_in_a は a 自身の
    # 全シート (a が miss していない限り) を基準に確定する (どの t を比較
    # 対象に含めても a 側の基準列がブレないようにするため)。
    class_cols: list[str] = (
        [f"wrong_in_{t}" for t in tags]
        + [f"fixed_by_{t}" for t in tags if t != "a"]
        + [f"new_regression_in_{t}" for t in tags if t != "a"]
    )
    class_lines = ["category,n_cells_gt," + ",".join(class_cols)]
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_sheet = {tag: {r["sheet_id"]: r for r in scores[tag]} for tag in tags}
    a_by_sheet = by_sheet["a"]

    # (3a) a 基準の n_cells_gt / wrong_in_a を a 自身の全シートで確定する。
    for a_row in a_by_sheet.values():
        for c in a_row.get("cells", []):
            cat = c["a_error_category"]
            by_cat[cat]["n_cells_gt"] += 1
            by_cat[cat]["wrong_in_a"] += int(not c["is_correct"])

    # (3b) t != "a" の wrong_in_t / fixed_by_t / new_regression_in_t は
    # 「a と t のペアワイズ共通シート」のみで確定する (他 tags の欠落と無関係)。
    for t in tags:
        if t == "a":
            continue
        pair_common = set(a_by_sheet) & set(by_sheet[t])
        n_pair_mismatch = len(set(a_by_sheet)) - len(pair_common)
        if n_pair_mismatch:
            print(f"  [警告] a/{t} 間で突合できたシート数が一致しない "
                  f"(共通{len(pair_common)}件、欠落{n_pair_mismatch}件) — "
                  f"{t} の再収集の欠落を確認すること")
        for sid in pair_common:
            a_cells = {(c["r"], c["c"]): c for c in a_by_sheet[sid].get("cells", [])}
            t_cells = {(c["r"], c["c"]): c for c in by_sheet[t][sid].get("cells", [])}
            for key, ac in a_cells.items():
                tc = t_cells.get(key)
                if tc is None:
                    continue
                cat = ac["a_error_category"]
                wrong_a = not ac["is_correct"]
                wrong_t = not tc["is_correct"]
                by_cat[cat][f"wrong_in_{t}"] += int(wrong_t)
                by_cat[cat][f"fixed_by_{t}"] += int(wrong_a and not wrong_t)
                by_cat[cat][f"new_regression_in_{t}"] += int(
                    (not wrong_a) and wrong_t
                )
    for cat in sorted(by_cat, key=lambda k: -by_cat[k]["n_cells_gt"]):
        v = by_cat[cat]
        class_lines.append(
            f"{cat},{v['n_cells_gt']}," + ",".join(str(v.get(col, 0)) for col in class_cols)
        )
        wrong_str = " ".join(f"{t}={v.get(f'wrong_in_{t}', 0)}" for t in tags)
        fix_str = " ".join(f"{t}={v.get(f'fixed_by_{t}', 0)}" for t in tags if t != "a")
        reg_str = " ".join(f"{t}={v.get(f'new_regression_in_{t}', 0)}" for t in tags if t != "a")
        print(f"  [{cat}] gt={v['n_cells_gt']} wrong: {wrong_str} | fixed: {fix_str} | new_regression: {reg_str}")
    (SCORING_DIR / "error_classification.csv").write_text("\n".join(class_lines) + "\n", encoding="utf-8")
    print("\n[done] 全出力: " + str(SCORING_DIR))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", choices=sorted(NPZ_DIRS))
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--compare-tags", nargs="+", default=None,
                     help="--compare で使うタグ列 (省略時は a c1 c2 c3 c の5列)")
    args = ap.parse_args()

    gts = _sc.load_ground_truth()
    print(f"[info] 有効盤面数: {len(gts)} (60 - not_a_board 5)")

    if args.score:
        score_one(args.score, gts)
        return
    if args.compare:
        tags = tuple(args.compare_tags) if args.compare_tags else BASELINE_TAGS
        compare_all(tags)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
