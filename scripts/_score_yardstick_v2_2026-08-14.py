"""物差し v2 (55有効盤面) 採点+3構成比較ドライバ (2026-08-15)。

## これは何か
data/verify/yardstick_v2_2026-08-14/labels_final_from_user.json (userレビュー
完了60盤面) を manifest.json の init_grid に適用して正解盤面を復元し、
以下3構成のSTABLE盤面npzと突合して一致率を採点する。

    構成a: 本番採用構成そのもの (収集済み npz、再収集不要)
    構成b: 構成a + --stable-majority-window
    構成c: 構成a + OJAMA_FALL系3フラグ

## 使い方
    python scripts/_score_yardstick_v2_2026-08-14.py --list-chunks   # 再収集対象チャンク一覧を出す
    python scripts/_score_yardstick_v2_2026-08-14.py --score a       # 構成aを採点
    python scripts/_score_yardstick_v2_2026-08-14.py --score b       # 構成bを採点 (再収集済み前提)
    python scripts/_score_yardstick_v2_2026-08-14.py --score c       # 構成cを採点 (再収集済み前提)
    python scripts/_score_yardstick_v2_2026-08-14.py --compare       # a/b/c比較表+分類表を出力
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
YARDSTICK_DIR = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14"
SCORING_DIR = YARDSTICK_DIR / "scoring"
MANIFEST_PATH = YARDSTICK_DIR / "manifest.json"
LABELS_PATH = YARDSTICK_DIR / "labels_final_from_user.json"

NPZ_DIR_A = _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_2026-08-14"
NPZ_DIR_B = _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_b_smw_2026-08-15"
NPZ_DIR_C = _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_c_ojamafall_2026-08-15"
NPZ_DIRS: dict[str, Path] = {"a": NPZ_DIR_A, "b": NPZ_DIR_B, "c": NPZ_DIR_C}

UNKNOWN_VALUE: int = 10  # 採点分母から除外する値 (labels_final_from_user.json の規約)
N_ROWS: int = 13  # 隠し段込みの行数 (row0=隠し段)
N_COLS: int = 6
# 突合フォールバックの許容ズレ [秒] (frame_idx完全一致が取れない場合の最近傍探索窓)
NEAREST_T_SEC_TOLERANCE: float = 0.35

WRONG_CELL_RE = re.compile(r"r(\d+)c(\d+)=(\d+)")


def load_ground_truth() -> list[dict[str, Any]]:
    """manifest.json + labels_final_from_user.json から正解盤面リストを作る。

    status=not_a_board は測定対象外として除外する (5枚)。
    """
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    label_by_id = {s["sheet_id"]: s for s in labels["sheets"]}
    out: list[dict[str, Any]] = []
    for entry in manifest:
        sid = entry["sheet_id"]
        lab = label_by_id.get(sid)
        if lab is None:
            raise ValueError(f"ラベル欠落 (fail-silent禁止, 要報告): {sid}")
        if lab["status"] == "not_a_board":
            continue
        grid = [row[:] for row in entry["init_grid"]]
        for wc in lab["wrong_cells"]:
            m = WRONG_CELL_RE.match(wc)
            if not m:
                raise ValueError(f"wrong_cellsパース失敗: {sid} {wc!r}")
            r, c, v = int(m[1]), int(m[2]), int(m[3])
            grid[r][c] = v
        out.append({
            "sheet_id": sid,
            "video_id": entry["video_id"],
            "side": entry["side"],
            "frame_idx": entry["frame_idx"],
            "t_sec": entry["t_sec"],
            "phase": entry["phase"],
            "fill_ratio": entry["fill_ratio"],
            "has_ojama": entry["has_ojama"],
            "init_grid": entry["init_grid"],
            "corrected_grid": grid,
        })
    return out


def load_npz_index(npz_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """npzディレクトリ全体を (video_id, side) -> レコードリスト の索引にする。"""
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for npz_path in sorted(npz_dir.glob("*.npz")):
        d = np.load(npz_path, allow_pickle=True)
        n = len(d["frame_idx"])
        for i in range(n):
            # npz内は "video_c109"/"_hold_video_c96" 形式、manifest側は "c109" 形式
            # (video_id_of() と同じ剥がし順序: 先に "_hold_" 、次に "video_")
            vid = str(d["video_id"][i]).removeprefix("_hold_").removeprefix("video_")
            side = str(d["side"][i])
            index[(vid, side)].append({
                "frame_idx": int(d["frame_idx"][i]),
                "t_sec": float(d["t_sec"][i]),
                "grid": d["grids"][i],
                "npz": npz_path.name,
            })
    return index


def match_record(
    gt: dict[str, Any], index: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str]:
    """正解盤面1件に対応するnpzレコードを探す。(record, method) を返す。

    method: "exact" (frame_idx完全一致) / "nearest" (±NEAREST_T_SEC_TOLERANCE秒フォールバック) / "miss"
    """
    key = (gt["video_id"], gt["side"])
    candidates = index.get(key, [])
    for rec in candidates:
        if rec["frame_idx"] == gt["frame_idx"]:
            return rec, "exact"
    best, best_dt = None, None
    for rec in candidates:
        dt = abs(rec["t_sec"] - gt["t_sec"])
        if best_dt is None or dt < best_dt:
            best, best_dt = rec, dt
    if best is not None and best_dt is not None and best_dt <= NEAREST_T_SEC_TOLERANCE:
        return best, "nearest"
    return None, "miss"


def cell_mask_and_match(corrected: list, pred: np.ndarray, include_row0: bool) -> tuple[np.ndarray, np.ndarray]:
    """採点対象マスクと一致配列を返す (both shape (13,6) bool)。"""
    corr = np.array(corrected, dtype=np.int64)
    mask = corr != UNKNOWN_VALUE
    if not include_row0:
        mask[0, :] = False
    match = (corr == pred) & mask
    return mask, match


def classify_error(init_val: int, correct_val: int) -> str:
    """誤りセル1件の分類ラベルを返す (構成aのinit_gridベース分類、タスク仕様の5分類)。"""
    COLORS = {1, 2, 3, 4, 5}
    if init_val == correct_val:
        return "no_error"
    if init_val == 0 and correct_val in COLORS:
        return "W9_dropout_empty_to_color"
    if init_val in COLORS and correct_val == 0:
        return "dropout_color_to_empty"
    if init_val == 9 and correct_val in COLORS:
        return "color_to_ojama"  # 013型: 本来色ぷよなのにおじゃまと誤認
    if init_val in COLORS and correct_val == 9:
        return "ojama_to_color"
    if {init_val, correct_val} == {1, 5}:
        return "W10_red_purple"
    if init_val in COLORS and correct_val in COLORS:
        return "other_color_confusion"
    if init_val == 0 and correct_val == 9:
        return "empty_to_ojama"
    if init_val == 9 and correct_val == 0:
        return "ojama_to_empty"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-chunks", action="store_true")
    ap.add_argument("--score", choices=["a", "b", "c"])
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    gts = load_ground_truth()
    print(f"[info] 有効盤面数: {len(gts)} (60 - not_a_board 5)")

    if args.list_chunks:
        _list_chunks(gts)
        return
    if args.score:
        _score_one(args.score, gts)
        return
    if args.compare:
        _compare_all(gts)
        return
    ap.print_help()


def _list_chunks(gts: list[dict[str, Any]]) -> None:
    """55盤面が属する (video_id, chunk_idx) を特定し、b/c再収集対象を出す。"""
    index_a = load_npz_index(NPZ_DIR_A)
    chunk_of: dict[tuple[str, str, int], str] = {}
    for npz_path in sorted(NPZ_DIR_A.glob("*.npz")):
        m = re.match(r"(.+)_chunk(\d+)\.npz", npz_path.name)
        vid, chunk = m[1], int(m[2])
        d = np.load(npz_path, allow_pickle=True)
        for i in range(len(d["frame_idx"])):
            npz_vid = str(d["video_id"][i]).removeprefix("_hold_").removeprefix("video_")
            chunk_of[(npz_vid, str(d["side"][i]), int(d["frame_idx"][i]))] = f"chunk{chunk}"
    needed: set[tuple[str, str]] = set()
    miss = 0
    for gt in gts:
        rec, method = match_record(gt, index_a)
        if rec is None:
            miss += 1
            continue
        chunk_key = chunk_of.get((gt["video_id"], gt["side"], rec["frame_idx"]))
        needed.add((gt["video_id"], rec["npz"]))
    print(f"[info] 突合失敗(miss): {miss}")
    print(f"[info] 再収集対象チャンク数: {len(needed)}")
    for vid, npz_name in sorted(needed):
        print(f"  {vid} -> {npz_name}")


def _score_one(tag: str, gts: list[dict[str, Any]]) -> None:
    """構成tagのnpzを採点し、CSVとJSON集計をSCORING_DIRに書き出す。"""
    npz_dir = NPZ_DIRS[tag]
    index = load_npz_index(npz_dir)
    rows = []
    for gt in gts:
        rec, method = match_record(gt, index)
        if rec is None:
            rows.append({**_row_meta(gt), "match_method": "miss"})
            continue
        mask_ex, match_ex = cell_mask_and_match(gt["corrected_grid"], rec["grid"], include_row0=False)
        mask_in, match_in = cell_mask_and_match(gt["corrected_grid"], rec["grid"], include_row0=True)
        cells = []  # 行0除く可視12行×6列の全セル (採点対象=correct!=UNKNOWN のみ)
        for r in range(1, N_ROWS):
            for c in range(N_COLS):
                if not mask_ex[r, c]:
                    continue
                init_val = int(np.array(gt["init_grid"])[r, c])
                corr_val = int(np.array(gt["corrected_grid"])[r, c])
                pred_val = int(rec["grid"][r, c])
                cells.append({
                    "r": r, "c": c, "init": init_val, "correct": corr_val, "pred": pred_val,
                    "is_correct": bool(match_ex[r, c]),
                    "a_error_category": classify_error(init_val, corr_val),
                })
        rows.append({
            **_row_meta(gt),
            "match_method": method,
            "n_cells_excl_row0": int(mask_ex.sum()),
            "n_correct_excl_row0": int(match_ex.sum()),
            "n_cells_incl_row0": int(mask_in.sum()),
            "n_correct_incl_row0": int(match_in.sum()),
            "cells": cells,
        })
    out_path = SCORING_DIR / f"score_{tag}.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    n_miss = sum(1 for r in rows if r["match_method"] == "miss")
    n_nearest = sum(1 for r in rows if r["match_method"] == "nearest")
    tot_cells = sum(r.get("n_cells_excl_row0", 0) for r in rows)
    tot_correct = sum(r.get("n_correct_excl_row0", 0) for r in rows)
    acc = tot_correct / tot_cells if tot_cells else float("nan")
    print(f"[score:{tag}] miss={n_miss} nearest_fallback={n_nearest} "
          f"cells(行0除)={tot_cells} correct={tot_correct} acc={acc:.4%}")
    print(f"[score:{tag}] -> {out_path}")


def _row_meta(gt: dict[str, Any]) -> dict[str, Any]:
    return {
        "sheet_id": gt["sheet_id"], "video_id": gt["video_id"], "side": gt["side"],
        "phase": gt["phase"], "fill_ratio": gt["fill_ratio"], "has_ojama": gt["has_ojama"],
    }


FILL_HIGH_THRESHOLD: float = 0.70  # 満杯・準満杯帯の閾値 (README記載のサンプリング方針と同値)


def _fill_band(fill_ratio: float) -> str:
    return "high(>=0.70)" if fill_ratio >= FILL_HIGH_THRESHOLD else "low(<0.70)"


def _load_scores() -> dict[str, list[dict[str, Any]]]:
    out = {}
    for tag in ("a", "b", "c"):
        path = SCORING_DIR / f"score_{tag}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"score_{tag}.json が無い。先に --score {tag} を実行すること (fail-silent禁止)")
        out[tag] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _natural_population_rates() -> dict[str, float]:
    """全収集候補 (npz-a 48チャンク全体) から has_ojama×fill_band の自然な構成比を推定する。

    旧物差し (board_labels_2026-07-31) 自体の構成比は構造化記録が残っていないため、
    無作為抽出時の自然母集団比の近似として扱う (本関数の返り値は近似値、報告時に明記必須)。
    """
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for npz_path in sorted(NPZ_DIR_A.glob("*.npz")):
        d = np.load(npz_path, allow_pickle=True)
        grids = d["grids"]
        for i in range(len(grids)):
            visible = grids[i][1:, :]  # 行0(隠し段)除く
            has_ojama = bool(np.any(visible == 9))
            fill_ratio = float(np.mean(visible != 0))
            band = _fill_band(fill_ratio)
            counts[f"ojama={has_ojama}|fill={band}"] += 1
            total += 1
    return {k: v / total for k, v in counts.items()} | {"_n_total_candidates": total}


def _stratum_key(gt_row: dict[str, Any]) -> str:
    return f"ojama={gt_row['has_ojama']}|fill={_fill_band(gt_row['fill_ratio'])}"


def _accuracy(rows: list[dict[str, Any]], key_correct: str, key_total: str) -> tuple[int, int, float]:
    tot = sum(r.get(key_total, 0) for r in rows)
    cor = sum(r.get(key_correct, 0) for r in rows)
    return cor, tot, (cor / tot if tot else float("nan"))


def _compare_all(gts: list[dict[str, Any]]) -> None:
    """3構成の比較表・分類表・層別再重み付け参考値をまとめて出力する。"""
    scores = _load_scores()
    SCORING_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== (1) 全体一致率 (行0除く/込む) ===")
    overall_lines = ["config,n_miss,n_nearest,cells_excl_row0,correct_excl_row0,acc_excl_row0,"
                      "cells_incl_row0,correct_incl_row0,acc_incl_row0"]
    for tag in ("a", "b", "c"):
        rows = scores[tag]
        n_miss = sum(1 for r in rows if r["match_method"] == "miss")
        n_nearest = sum(1 for r in rows if r["match_method"] == "nearest")
        c_ex, t_ex, acc_ex = _accuracy(rows, "n_correct_excl_row0", "n_cells_excl_row0")
        c_in, t_in, acc_in = _accuracy(rows, "n_correct_incl_row0", "n_cells_incl_row0")
        print(f"  [{tag}] miss={n_miss} nearest={n_nearest} "
              f"行0除={acc_ex:.4%}({c_ex}/{t_ex})  行0込[参考,人手未検証]={acc_in:.4%}({c_in}/{t_in})")
        overall_lines.append(f"{tag},{n_miss},{n_nearest},{t_ex},{c_ex},{acc_ex:.6f},{t_in},{c_in},{acc_in:.6f}")
    (SCORING_DIR / "overall_accuracy.csv").write_text("\n".join(overall_lines) + "\n", encoding="utf-8")

    # a/b/c で突合できたシートが異なるため、denominatorが構成間で不揃い。
    # 公平比較のため「3構成全てで突合成功したシートのみ」の共通部分集合でも算出する。
    common_ids = (
        {r["sheet_id"] for r in scores["a"] if r["match_method"] != "miss"}
        & {r["sheet_id"] for r in scores["b"] if r["match_method"] != "miss"}
        & {r["sheet_id"] for r in scores["c"] if r["match_method"] != "miss"}
    )
    print(f"\n=== (1b) 3構成共通突合部分集合での一致率 (公平比較、n={len(common_ids)}/55) ===")
    common_lines = ["config,cells_excl_row0,correct_excl_row0,acc_excl_row0"]
    for tag in ("a", "b", "c"):
        sub = [r for r in scores[tag] if r["sheet_id"] in common_ids]
        c_ex, t_ex, acc_ex = _accuracy(sub, "n_correct_excl_row0", "n_cells_excl_row0")
        print(f"  [{tag}] 行0除={acc_ex:.4%}({c_ex}/{t_ex})")
        common_lines.append(f"{tag},{t_ex},{c_ex},{acc_ex:.6f}")
    (SCORING_DIR / "common_subset_accuracy.csv").write_text("\n".join(common_lines) + "\n", encoding="utf-8")

    print("\n=== (2) 層別一致率 (行0除く) ===")
    strat_lines = ["axis,stratum,config,cells,correct,accuracy"]
    for axis_name, axis_fn in (
        ("phase", lambda r: r["phase"]),
        ("has_ojama", lambda r: str(r["has_ojama"])),
        ("fill_band", lambda r: _fill_band(r["fill_ratio"])),
        ("video_id", lambda r: r["video_id"]),
    ):
        strata = sorted({axis_fn(r) for r in scores["a"]})
        for st in strata:
            for tag in ("a", "b", "c"):
                sub = [r for r in scores[tag] if axis_fn(r) == st]
                c_ex, t_ex, acc_ex = _accuracy(sub, "n_correct_excl_row0", "n_cells_excl_row0")
                strat_lines.append(f"{axis_name},{st},{tag},{t_ex},{c_ex},{acc_ex:.6f}")
            print(f"  [{axis_name}={st}] "
                  + " / ".join(f"{tag}={_accuracy([r for r in scores[tag] if axis_fn(r) == st], 'n_correct_excl_row0', 'n_cells_excl_row0')[2]:.3%}"
                                for tag in ("a", "b", "c")))
    (SCORING_DIR / "stratified_accuracy.csv").write_text("\n".join(strat_lines) + "\n", encoding="utf-8")

    print("\n=== (3) 誤り分類表 (構成aのinit_grid基準の固定カテゴリ、b/cで解消したか) ===")
    class_lines = ["category,n_cells_gt,wrong_in_a,wrong_in_b,wrong_in_c,"
                    "fixed_by_b,fixed_by_c,new_regression_in_b,new_regression_in_c"]
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_sheets_scored = min(len(scores["a"]), len(scores["b"]), len(scores["c"]))
    a_by_sheet = {r["sheet_id"]: r for r in scores["a"]}
    b_by_sheet = {r["sheet_id"]: r for r in scores["b"]}
    c_by_sheet = {r["sheet_id"]: r for r in scores["c"]}
    common_sheets = set(a_by_sheet) & set(b_by_sheet) & set(c_by_sheet)
    n_sheet_mismatch = len(set(a_by_sheet)) - len(common_sheets)
    if n_sheet_mismatch:
        print(f"  [警告] a/b/c間で突合できたシート数が一致しない (共通{len(common_sheets)}件、"
              f"a単独{n_sheet_mismatch}件) — b/c再収集の欠落を確認すること")
    for sid in common_sheets:
        a_cells = {(c["r"], c["c"]): c for c in a_by_sheet[sid].get("cells", [])}
        b_cells = {(c["r"], c["c"]): c for c in b_by_sheet[sid].get("cells", [])}
        c_cells = {(c["r"], c["c"]): c for c in c_by_sheet[sid].get("cells", [])}
        for key, ac in a_cells.items():
            bc = b_cells.get(key)
            cc = c_cells.get(key)
            if bc is None or cc is None:
                continue  # miss (突合失敗シート) は分類対象外
            cat = ac["a_error_category"]
            by_cat[cat]["n_cells_gt"] += 1
            wrong_a = not ac["is_correct"]
            wrong_b = not bc["is_correct"]
            wrong_c = not cc["is_correct"]
            by_cat[cat]["wrong_in_a"] += int(wrong_a)
            by_cat[cat]["wrong_in_b"] += int(wrong_b)
            by_cat[cat]["wrong_in_c"] += int(wrong_c)
            by_cat[cat]["fixed_by_b"] += int(wrong_a and not wrong_b)
            by_cat[cat]["fixed_by_c"] += int(wrong_a and not wrong_c)
            by_cat[cat]["new_regression_in_b"] += int((not wrong_a) and wrong_b)
            by_cat[cat]["new_regression_in_c"] += int((not wrong_a) and wrong_c)
    for cat in sorted(by_cat, key=lambda k: -by_cat[k]["n_cells_gt"]):
        v = by_cat[cat]
        class_lines.append(f"{cat},{v['n_cells_gt']},{v['wrong_in_a']},{v['wrong_in_b']},{v['wrong_in_c']},"
                            f"{v['fixed_by_b']},{v['fixed_by_c']},{v['new_regression_in_b']},{v['new_regression_in_c']}")
        print(f"  [{cat}] gt={v['n_cells_gt']} wrong: a={v['wrong_in_a']} b={v['wrong_in_b']} c={v['wrong_in_c']} | "
              f"fixed: b={v['fixed_by_b']} c={v['fixed_by_c']} | new_regression: b={v['new_regression_in_b']} c={v['new_regression_in_c']}")
    (SCORING_DIR / "error_classification.csv").write_text("\n".join(class_lines) + "\n", encoding="utf-8")

    print("\n=== (4) 自然母集団比で再重み付けした参考値 (近似値、旧物差し自体の構成比ではない) ===")
    pop_rates = _natural_population_rates()
    n_pop = pop_rates.pop("_n_total_candidates")
    print(f"  母集団候補フレーム数: {n_pop} (npz-a 全48チャンク由来、has_ojama×fill_band 構成)")
    reweighted_lines = ["config,stratum,pop_weight,sample_cells,sample_correct,sample_accuracy,weighted_contribution"]
    for tag in ("a", "b", "c"):
        acc_sum = 0.0
        weight_sum_used = 0.0
        missing_strata = []
        for stratum, weight in pop_rates.items():
            sub = [r for r in scores[tag] if _stratum_key(r) == stratum]
            c_ex, t_ex, acc_ex = _accuracy(sub, "n_correct_excl_row0", "n_cells_excl_row0")
            if t_ex == 0:
                missing_strata.append(stratum)
                continue
            acc_sum += weight * acc_ex
            weight_sum_used += weight
            reweighted_lines.append(f"{tag},{stratum},{weight:.4f},{t_ex},{c_ex},{acc_ex:.6f},{weight*acc_ex:.6f}")
        reweighted_acc = acc_sum / weight_sum_used if weight_sum_used else float("nan")
        print(f"  [{tag}] 再重み付け一致率(参考)={reweighted_acc:.4%} "
              f"(重み補正カバー率={weight_sum_used:.1%}"
              + (f", 標本ゼロ層={missing_strata}" if missing_strata else "") + ")")
    (SCORING_DIR / "reweighted_accuracy.csv").write_text("\n".join(reweighted_lines) + "\n", encoding="utf-8")
    print("\n[done] 全出力: " + str(SCORING_DIR))


if __name__ == "__main__":
    main()
