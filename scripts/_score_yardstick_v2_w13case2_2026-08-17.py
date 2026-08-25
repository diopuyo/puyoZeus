"""W13根治 案2 (patch-NCC HSV ANDガード) の物差しv2採点 + 4構成比較 (2026-08-17)。

`scripts/_score_yardstick_v2_2026-08-14.py` の採点ロジック (突合/セル一致判定/
誤り分類) をそのまま再利用し (ファイル名にハイフンを含むため importlib で動的
import)、以下4構成を55有効盤面で比較する:

    a(=c1p)  : 現行本番採用構成そのもの (data/verify/.../scoring_ablation/score_c1p.json 流用)
    w13      : a + --enable-highlight-override (案1、既存npz流用)
    w13p2    : a + --enable-patch-fp-hsv-guard (案2、本タスクで新規収集)
    w13both  : a + 両方併用 (案1+2、本タスクで新規収集)

W16教訓 (測定器事故) に基づき、2段階の公平化を行う:
    stage1: 4構成全てで突合成功 (match_method != miss) した共通シートのみ
    stage2: 4構成全てで match_method == "exact" (frame_idx完全一致、
            ±0.35秒フォールバック無し) だった共通シートのみ (= 同一フレーム限定、
            STABLE間引き周期の差による分母ズレを排除)

出力は data/verify/yardstick_v2_2026-08-14/scoring_ablation/ に書く。

使い方:
    python scripts/_score_yardstick_v2_w13case2_2026-08-17.py --score w13p2
    python scripts/_score_yardstick_v2_w13case2_2026-08-17.py --score w13both
    python scripts/_score_yardstick_v2_w13case2_2026-08-17.py --compare
    python scripts/_score_yardstick_v2_w13case2_2026-08-17.py --check-regressed-cells
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

TAGS: tuple[str, ...] = ("c1p", "w13", "w13p2", "w13both")
TAG_LABEL: dict[str, str] = {
    "c1p": "ベース (現行本番採用構成)",
    "w13": "案1 (highlight_override)",
    "w13p2": "案2 (patch_fp_hsv_guard)",
    "w13both": "案1+2 併用",
}

NPZ_DIRS: dict[str, Path] = {
    "c1p": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_c1p_fix_2026-08-15",
    "w13": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_w13_2026-08-16",
    "w13p2": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_w13p2_2026-08-17",
    "w13both": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_w13both_2026-08-17",
}

def score_one(tag: str) -> None:
    """構成tagのnpzを採点し、score_{tag}.json を scoring_ablation/ に書く。"""
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
    n_exact = sum(1 for r in rows if r["match_method"] == "exact")
    tot = sum(r.get("n_cells_excl_row0", 0) for r in rows)
    cor = sum(r.get("n_correct_excl_row0", 0) for r in rows)
    acc = cor / tot if tot else float("nan")
    print(f"[score:{tag}] miss={n_miss} nearest={n_nearest} exact={n_exact} "
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


def _direction_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    """cells の a_error_category から方向別 (空→ぷよ有 / ぷよ有→空) の誤り件数を集計。"""
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        for c in r.get("cells", []):
            cat = c["a_error_category"]
            if cat == "no_error":
                continue
            if c["init"] == 0 and c["correct"] != 0:
                direction = "empty_to_puyo(正解は空でない)"
            elif c["init"] != 0 and c["correct"] == 0:
                direction = "puyo_to_empty(正解は空)"
            else:
                direction = "other"
            counts[f"{direction}|wrong_now={not c['is_correct']}"] += 1
    return dict(counts)


def compare_all() -> None:
    """4構成 (c1p/w13/w13p2/w13both) の2段階公平化比較を出力する。"""
    scores = _load_scores(TAGS)
    SCORING_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== (0) 生の全体一致率 (突合ズレ込み、参考値) ===")
    for tag in TAGS:
        rows = scores[tag]
        n_miss = sum(1 for r in rows if r["match_method"] == "miss")
        n_nearest = sum(1 for r in rows if r["match_method"] == "nearest")
        n_exact = sum(1 for r in rows if r["match_method"] == "exact")
        c_ex, t_ex, acc_ex = _sc._accuracy(rows, "n_correct_excl_row0", "n_cells_excl_row0")
        print(f"  [{tag}:{TAG_LABEL[tag]}] miss={n_miss} nearest={n_nearest} exact={n_exact} "
              f"acc={acc_ex:.4%}({c_ex}/{t_ex})")

    by_sheet = {tag: {r["sheet_id"]: r for r in scores[tag]} for tag in TAGS}

    # stage1: 4構成全てで突合成功 (match_method != miss)
    stage1_ids = set(by_sheet["c1p"])
    for tag in TAGS:
        stage1_ids &= {sid for sid, r in by_sheet[tag].items() if r["match_method"] != "miss"}
    print(f"\n=== (1) stage1: 4構成共通突合部分集合での一致率 (n={len(stage1_ids)}/55) ===")
    stage1_lines = ["config,cells_excl_row0,correct_excl_row0,acc_excl_row0"]
    for tag in TAGS:
        sub = [by_sheet[tag][sid] for sid in stage1_ids]
        c_ex, t_ex, acc_ex = _sc._accuracy(sub, "n_correct_excl_row0", "n_cells_excl_row0")
        print(f"  [{tag}:{TAG_LABEL[tag]}] 行0除={acc_ex:.4%}({c_ex}/{t_ex})")
        stage1_lines.append(f"{tag},{t_ex},{c_ex},{acc_ex:.6f}")
    (SCORING_DIR / "w13case2_stage1_common_subset.csv").write_text(
        "\n".join(stage1_lines) + "\n", encoding="utf-8")

    # stage2: 4構成全てで match_method == "exact" (同一フレーム限定)
    stage2_ids = {
        sid for sid in stage1_ids
        if all(by_sheet[tag][sid]["match_method"] == "exact" for tag in TAGS)
    }
    print(f"\n=== (2) stage2: 同一フレーム限定 (match_method==exact 共通、n={len(stage2_ids)}/55) ===")
    stage2_lines = ["config,cells_excl_row0,correct_excl_row0,acc_excl_row0"]
    for tag in TAGS:
        sub = [by_sheet[tag][sid] for sid in stage2_ids]
        c_ex, t_ex, acc_ex = _sc._accuracy(sub, "n_correct_excl_row0", "n_cells_excl_row0")
        print(f"  [{tag}:{TAG_LABEL[tag]}] 行0除={acc_ex:.4%}({c_ex}/{t_ex})")
        stage2_lines.append(f"{tag},{t_ex},{c_ex},{acc_ex:.6f}")
    (SCORING_DIR / "w13case2_stage2_same_frame.csv").write_text(
        "\n".join(stage2_lines) + "\n", encoding="utf-8")

    print("\n=== (3) 方向別内訳 (stage2 部分集合、空→ぷよ有り / ぷよ有り→空) ===")
    for tag in TAGS:
        sub = [by_sheet[tag][sid] for sid in stage2_ids]
        breakdown = _direction_breakdown(sub)
        print(f"  [{tag}:{TAG_LABEL[tag]}] {breakdown}")

    print("\n=== (4) 誤り分類表 (stage2部分集合、c1p基準カテゴリ別 wrong 件数) ===")
    class_lines = ["category,n_cells_gt"] + [f"wrong_in_{t}" for t in TAGS]
    class_lines = [",".join(class_lines)]
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sid in stage2_ids:
        c1p_cells = {(c["r"], c["c"]): c for c in by_sheet["c1p"][sid].get("cells", [])}
        tag_cells = {tag: {(c["r"], c["c"]): c for c in by_sheet[tag][sid].get("cells", [])} for tag in TAGS}
        for key, base_c in c1p_cells.items():
            cat = base_c["a_error_category"]
            by_cat[cat]["n_cells_gt"] += 1
            for tag in TAGS:
                tc = tag_cells[tag].get(key)
                if tc is None:
                    continue
                by_cat[cat][f"wrong_in_{tag}"] += int(not tc["is_correct"])
    for cat in sorted(by_cat, key=lambda k: -by_cat[k]["n_cells_gt"]):
        v = by_cat[cat]
        class_lines.append(
            f"{cat},{v['n_cells_gt']}," + ",".join(str(v.get(f"wrong_in_{t}", 0)) for t in TAGS)
        )
        wrong_str = " ".join(f"{t}={v.get(f'wrong_in_{t}', 0)}" for t in TAGS)
        print(f"  [{cat}] gt={v['n_cells_gt']} wrong: {wrong_str}")
    (SCORING_DIR / "w13case2_error_classification.csv").write_text(
        "\n".join(class_lines) + "\n", encoding="utf-8")
    print("\n[done] 全出力: " + str(SCORING_DIR))


def check_regressed_cells() -> None:
    """案1(w13)で悪化した13セル (000_c109_1P_f652064 9セル / 002_c11_2P_f54124 4セル) が
    案2(w13p2)/案1+2(w13both) でどうなるかを個別出力する。
    """
    scores = _load_scores(("c1p", "w13", "w13p2", "w13both"))
    by_sheet = {tag: {r["sheet_id"]: r for r in scores[tag]} for tag in scores}
    target_sheets = ("000_c109_1P_f652064", "002_c11_2P_f54124")
    for sid in target_sheets:
        print(f"\n=== {sid} ===")
        cells_by_tag = {}
        for tag in ("c1p", "w13", "w13p2", "w13both"):
            row = by_sheet[tag].get(sid)
            if row is None:
                print(f"  [{tag}] シートなし")
                continue
            cells_by_tag[tag] = {(c["r"], c["c"]): c for c in row.get("cells", [])}
        # c1p は正解、w13 は不正解 だったセルのみ抽出 (= 案1の新規悪化13セル)
        base = cells_by_tag.get("c1p", {})
        w13c = cells_by_tag.get("w13", {})
        regressed_keys = [
            key for key, bc in base.items()
            if bc["is_correct"] and key in w13c and not w13c[key]["is_correct"]
        ]
        print(f"  案1で悪化したセル数: {len(regressed_keys)}")
        for key in sorted(regressed_keys):
            r, c = key
            line = f"  r{r}c{c}: correct={base[key]['correct']}"
            for tag in ("c1p", "w13", "w13p2", "w13both"):
                cd = cells_by_tag.get(tag, {}).get(key)
                if cd is None:
                    line += f" | {tag}=なし"
                else:
                    mark = "OK" if cd["is_correct"] else f"NG(pred={cd['pred']})"
                    line += f" | {tag}={mark}"
            print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", choices=["w13p2", "w13both"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--check-regressed-cells", action="store_true")
    args = ap.parse_args()

    if args.score:
        score_one(args.score)
        return
    if args.compare:
        compare_all()
        return
    if args.check_regressed_cells:
        check_regressed_cells()
        return
    ap.print_help()


if __name__ == "__main__":
    main()
