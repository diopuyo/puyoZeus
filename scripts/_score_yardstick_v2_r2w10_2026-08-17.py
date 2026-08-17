"""認識強化統一測定 (2026-08-17): R2/W10ガードの物差しv2採点 + 4構成比較。

`scripts/_score_yardstick_v2_2026-08-14.py` の採点ロジックをそのまま再利用し
(ファイル名にハイフンを含むため importlib で動的 import)、以下4構成を55有効
盤面で比較する:

    A: 現行本番採用構成そのもの (= score_w13p2.json 流用、再収集不要)
    B: A + --enable-floating-gap-restore (R2)
    C: A + --enable-landing-color-guard (W10ガード)
    D: A + 両方

W16教訓 (測定器事故) に基づき、2段階の公平化を行う:
    stage1: 4構成全てで突合成功 (match_method != miss) した共通シートのみ
    stage2: 4構成全てで match_method == "exact" だった共通シートのみ
            (同一フレーム限定、STABLE間引き周期の差による分母ズレを排除)

使い方:
    python scripts/_score_yardstick_v2_r2w10_2026-08-17.py --score b
    python scripts/_score_yardstick_v2_r2w10_2026-08-17.py --score c
    python scripts/_score_yardstick_v2_r2w10_2026-08-17.py --score d
    python scripts/_score_yardstick_v2_r2w10_2026-08-17.py --compare
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

TAGS: tuple[str, ...] = ("a", "b", "c", "d", "e", "f")
TAG_LABEL: dict[str, str] = {
    "a": "A: 採用済みベース (w13p2相当)",
    "b": "B: A + R2 (floating_gap_restore)",
    "c": "C: A + W10ガード (landing_color_guard)",
    "d": "D: A + R2 + W10ガード (併用)",
    "e": "E: D + 持続誤認26件修正2点 (override_color_guard + ojama_column_stack_fix)",
    "f": "F: E + W23根治 (next_history_starvation_fix)",
}

# 構成Aのnpz実体は再収集不要 (score_w13p2.json 収集時の yardstick_v2_boards_w13p2_
# 2026-08-17 をそのまま再利用)。B/C/Dの npz は本タスクで新規収集する。
NPZ_DIRS: dict[str, Path] = {
    "a": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_w13p2_2026-08-17",
    "b": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_r2_2026-08-17",
    "c": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_w10guard_2026-08-17",
    "d": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_r2w10_2026-08-17",
    "e": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_e_2026-08-17",
    "f": _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_f_2026-08-17",
}


def _ensure_score_a() -> None:
    """score_a.json を score_one("a") で再生成する (npz/frame_idx/t_sec 付き)。

    既存 score_w13p2.json は W16教訓以前のフォーマット (npz/frame_idx/t_sec
    列が無い) のため、持続誤認診断 (_diag_persistent_misread_2026-08-17.py)
    に必要な情報を含めて score_one 経由で作り直す。npz 実体
    (yardstick_v2_boards_w13p2_2026-08-17) は再収集不要 (既存資産を再スコアする
    だけで、収集自体は行わない)。
    """
    score_one("a")


def score_one(tag: str) -> None:
    """構成tag (b/c/d) のnpzを採点し、score_{tag}.json を書く。"""
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
            "npz": rec["npz"],
            "frame_idx": rec["frame_idx"],
            "t_sec": rec["t_sec"],
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


def compare_all() -> None:
    """4構成 (A/B/C/D) の2段階公平化比較を出力する。"""
    _ensure_score_a()
    scores = _load_scores(TAGS)
    SCORING_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== (0) 生の全体一致率 (突合ズレ込み、参考値) ===")
    for tag in TAGS:
        rows = scores[tag]
        n_miss = sum(1 for r in rows if r["match_method"] == "miss")
        n_nearest = sum(1 for r in rows if r["match_method"] == "nearest")
        n_exact = sum(1 for r in rows if r["match_method"] == "exact")
        c_ex, t_ex, acc_ex = _sc._accuracy(rows, "n_correct_excl_row0", "n_cells_excl_row0")
        # 盤面完全一致率 (誤り0セルの盤面割合) + 誤り1セルのみ/2セル以上の内訳
        n_full = sum(
            1 for r in rows
            if r["match_method"] != "miss" and r["n_correct_excl_row0"] == r["n_cells_excl_row0"]
        )
        n_one_wrong = sum(
            1 for r in rows
            if r["match_method"] != "miss"
            and r["n_cells_excl_row0"] - r["n_correct_excl_row0"] == 1
        )
        n_multi_wrong = sum(
            1 for r in rows
            if r["match_method"] != "miss"
            and r["n_cells_excl_row0"] - r["n_correct_excl_row0"] >= 2
        )
        n_scored = sum(1 for r in rows if r["match_method"] != "miss")
        print(f"  [{tag}:{TAG_LABEL[tag]}] miss={n_miss} nearest={n_nearest} exact={n_exact} "
              f"acc={acc_ex:.4%}({c_ex}/{t_ex}) "
              f"盤面完全一致={n_full}/{n_scored} 誤り1セル={n_one_wrong} 誤り2+セル={n_multi_wrong}")

    by_sheet = {tag: {r["sheet_id"]: r for r in scores[tag]} for tag in TAGS}

    stage1_ids = set(by_sheet["a"])
    for tag in TAGS:
        stage1_ids &= {sid for sid, r in by_sheet[tag].items() if r["match_method"] != "miss"}
    print(f"\n=== (1) stage1: 4構成共通突合部分集合での一致率 (n={len(stage1_ids)}/55) ===")
    stage1_lines = ["config,cells_excl_row0,correct_excl_row0,acc_excl_row0"]
    for tag in TAGS:
        sub = [by_sheet[tag][sid] for sid in stage1_ids]
        c_ex, t_ex, acc_ex = _sc._accuracy(sub, "n_correct_excl_row0", "n_cells_excl_row0")
        print(f"  [{tag}:{TAG_LABEL[tag]}] 行0除={acc_ex:.4%}({c_ex}/{t_ex})")
        stage1_lines.append(f"{tag},{t_ex},{c_ex},{acc_ex:.6f}")
    (SCORING_DIR / "r2w10_stage1_common_subset.csv").write_text(
        "\n".join(stage1_lines) + "\n", encoding="utf-8")

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
    (SCORING_DIR / "r2w10_stage2_same_frame.csv").write_text(
        "\n".join(stage2_lines) + "\n", encoding="utf-8")

    print("\n=== (3) 誤り分類表 (stage2部分集合、a基準カテゴリ別 wrong 件数) ===")
    class_lines = ["category,n_cells_gt"] + [f"wrong_in_{t}" for t in TAGS]
    class_lines = [",".join(class_lines)]
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sid in stage2_ids:
        a_cells = {(c["r"], c["c"]): c for c in by_sheet["a"][sid].get("cells", [])}
        tag_cells = {tag: {(c["r"], c["c"]): c for c in by_sheet[tag][sid].get("cells", [])} for tag in TAGS}
        for key, base_c in a_cells.items():
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
    (SCORING_DIR / "r2w10_error_classification.csv").write_text(
        "\n".join(class_lines) + "\n", encoding="utf-8")
    print("\n[done] 全出力: " + str(SCORING_DIR))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", choices=["a", "b", "c", "d", "e", "f"])
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
