"""持続誤認測定器 W24対策: 区間中の真値変化の再検証 (測定器事故10件目、2026-08-18)。

## 背景 (W24、docs/KNOWN_WEAKNESSES.md 参照)
`scripts/_diag_persistent_misread_2026-08-17.py::_find_run` は「アンカーframeで
観測した誤り値と一致する confirmed_board の連続区間長」を測るだけで、区間中に
本物のゲーム進行 (新規設置・おじゃま着弾・連鎖解決) でアンカー時点のラベルが
その先も正解であり続ける保証を一度も検証しない。c21系7件はこの誤集計で
「持続誤認」に誤計上されていた (書き込みframe計装+実画面照合で判明、
`scripts/_diag_c21_burst_2026-08-17.py`)。

## 対策の仕組み
npz には既に `tsumo_count` (自分側のツモ設置累計) が収録されている。これは
「本物のゲーム進行の物理時計」として使える — 新規設置が起きない限り値は
変わらない。区間 [lo, hi] のうち、アンカー frame の `tsumo_count` と完全一致する
アンカー隣接の最大連続部分区間だけを "truth_stable" 区間として採用し、
そこで測り直した持続フレーム数で判定し直す (区間分割 + 除外)。

区間内で tsumo_count が一切変化しない (n_transitions == 0) 場合は旧測定器と
完全に同じ結果になる (bit-identical、退行なし)。1回でも変化があれば
`truth_may_have_changed=True` を立て、truth_stable 側の値で再判定する。

## 意図的に保守的な設計 (fail-silent警戒)
tsumo_count が1回でも変化した区間は「アンカー単発ラベルの信頼区間外」として
扱い、truth_stable 側が閾値未満なら persistent から除外する。これにより
「1回だけ設置イベントに近接した短い区間」(例: W25雲の一部) も除外され得るが、
`n_tsumo_transitions_in_original_run` を全件に付与するため、人間が事後に
「1回だけの近接イベントなら復帰させてよい」と判断する材料は失わない
(数値だけで機械的に握りつぶさない)。

## 既存出力との関係 (新旧比較のため上書きしない)
`scripts/_diag_persistent_misread_2026-08-17.py::analyze_tag` の出力
(`data/verify/recognition_unified_2026-08-17/persistent_misread_{tag}.json`) は
一切変更しない。本スクリプトは同ディレクトリに
`persistent_misread_{tag}_truth_rechecked.json` を新規出力し、
`n_persistent_original` (旧測定器の値、無変更) と
`n_persistent_truth_verified` (新測定器の値) を並べて比較できるようにする。

使い方:
    python scripts/_diag_persistent_misread_truth_recheck_2026-08-18.py --tag f
    python scripts/_diag_persistent_misread_truth_recheck_2026-08-18.py --all
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_persist = importlib.import_module("scripts._diag_persistent_misread_2026-08-17")

# 旧測定器のOUT_DIRをそのまま使う (同じ場所に *_truth_rechecked.json を追加出力、
# 旧 persistent_misread_{tag}.json は一切上書きしない)。
OUT_DIR = _persist.OUT_DIR
PERSIST_FRAME_THRESHOLD = _persist.PERSIST_FRAME_THRESHOLD
EFFECTIVE_FPS = _persist.EFFECTIVE_FPS

# 区間内でこのフィールドが変化しない場合のみ「アンカー単発ラベルが区間全体の
# 正解として信用できる」とみなす。おじゃま着弾は受け側ツモ設置がgateなので
# (reference_ojama_landing_gated_by_placement)、tsumo_count 一本で
# 「新規設置・おじゃま着弾・それに伴う連鎖解決」の全イベントを捕捉できる。
TRUTH_CLOCK_FIELD: str = "tsumo_count"

# 除外判定は0-toleranceで一律に行うが (=1回でも設置イベントを跨いだら
# truth_stable側で再判定)、機械的な二値判定だけで握りつぶさないよう
# 遷移回数から3段階の確信度タグを付与する。実データでの分離点の目安
# (2026-08-18実測、data/verify/recognition_unified_2026-08-17/):
#   - W24確定 (c21系7件、認識は正しかった) は遷移8〜19回
#   - W25雲の隣接候補 (未確定、要人手レビュー) は遷移1〜2回
# の間に明確な断絶があったため、閾値をその中間に置く。
HIGH_CONFIDENCE_TRANSITION_COUNT: int = 3


def _confidence_tier(n_transitions: int) -> str:
    """遷移回数から確信度タグを返す (除外可否そのものではなく、人手レビューの
    優先度づけ用の付加情報)。"""
    if n_transitions == 0:
        return "no_transition"  # 旧測定器と完全に同じ判定 (bit-identical)
    if n_transitions < HIGH_CONFIDENCE_TRANSITION_COUNT:
        return "ambiguous"  # 要人手レビュー (W25雲等の近接単発イベントの疑いあり)
    return "high_confidence_artifact"  # W24型誤集計として高確信度


def _load_chunk_series_with_clock(npz_path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """旧 `_load_chunk_series` に `tsumo_count` 列を足しただけの拡張版。

    旧モジュールの関数は変更せず (新旧比較の健全性のため)、ここで別途ロードする。
    """
    import numpy as np

    d = np.load(npz_path, allow_pickle=True)
    n = len(d["frame_idx"])
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for i in range(n):
        vid = str(d["video_id"][i]).removeprefix("_hold_").removeprefix("video_")
        side = str(d["side"][i])
        by_key.setdefault((vid, side), []).append({
            "frame_idx": int(d["frame_idx"][i]),
            "t_sec": float(d["t_sec"][i]),
            "grid": d["grids"][i],
            TRUTH_CLOCK_FIELD: int(d[TRUTH_CLOCK_FIELD][i]),
        })
    for rows in by_key.values():
        rows.sort(key=lambda r: r["frame_idx"])
    return by_key


def truth_stable_bounds(
    rows: list[dict[str, Any]], lo: int, hi: int, anchor_idx: int,
) -> tuple[int, int, int]:
    """[lo, hi] のうち `TRUTH_CLOCK_FIELD` がアンカーと完全一致するアンカー隣接の
    最大連続部分区間 (stable_lo, stable_hi) と、[lo, hi] 全体で観測された
    ユニーク値数-1 (=遷移回数、n_transitions) を返す。

    n_transitions == 0 なら区間中ずっと同じ物理時計値 (=本物のゲーム進行なし)
    であり、旧測定器の判定をそのまま信用してよい。1以上なら区間中に新規設置
    イベントが起きており、アンカー単発ラベルを区間全体の正解に使うのはW24の
    誤集計パターンに該当し得る。
    """
    anchor_val = rows[anchor_idx][TRUTH_CLOCK_FIELD]
    stable_lo = anchor_idx
    while stable_lo - 1 >= lo and rows[stable_lo - 1][TRUTH_CLOCK_FIELD] == anchor_val:
        stable_lo -= 1
    stable_hi = anchor_idx
    while stable_hi + 1 <= hi and rows[stable_hi + 1][TRUTH_CLOCK_FIELD] == anchor_val:
        stable_hi += 1
    n_transitions = len({rows[i][TRUTH_CLOCK_FIELD] for i in range(lo, hi + 1)}) - 1
    return stable_lo, stable_hi, n_transitions


def _recheck_entry(
    rows: list[dict[str, Any]], anchor_idx: int, r: int, c: int, entry: dict[str, Any],
) -> dict[str, Any]:
    """旧エントリ (persistent_cells の1件) に truth-recheck 情報を付与して返す。

    lo_idx/hi_idx は旧 `_find_run` の結果から再取得する (旧エントリ自体には
    保存されていないため、同じ入力で `_find_run` を再実行する。値一致ロジック
    は不変なので旧集計とbit-identicalな lo/hi が得られる)。
    """
    run = _persist._find_run(rows, anchor_idx, r, c)
    lo, hi = run["lo_idx"], run["hi_idx"]
    stable_lo, stable_hi, n_transitions = truth_stable_bounds(rows, lo, hi, anchor_idx)
    stable_duration = rows[stable_hi]["t_sec"] - rows[stable_lo]["t_sec"]
    stable_frames_equiv = round(stable_duration * EFFECTIVE_FPS, 2)
    truth_may_have_changed = n_transitions > 0
    reclassified_non_persistent = stable_frames_equiv < PERSIST_FRAME_THRESHOLD
    return {
        **entry,
        "original_frames_equiv": entry["frames_equiv"],
        "original_duration_sec": entry["duration_sec"],
        "truth_stable_frames_equiv": stable_frames_equiv,
        "truth_stable_duration_sec": round(stable_duration, 4),
        "n_tsumo_transitions_in_original_run": n_transitions,
        "truth_may_have_changed": truth_may_have_changed,
        "reclassified_non_persistent": reclassified_non_persistent,
        "confidence_tier": _confidence_tier(n_transitions),
    }


def recheck_tag(tag: str) -> dict[str, Any]:
    """`persistent_misread_{tag}.json` (旧測定器の確定出力) を読み、
    persistent_cells 全件に truth-recheck を適用した結果を返す。"""
    old_path = OUT_DIR / f"persistent_misread_{tag}.json"
    if not old_path.exists():
        raise FileNotFoundError(
            f"{old_path} が無い (先に _diag_persistent_misread_2026-08-17.py "
            f"--tag {tag} を実行すること、fail-silent禁止で要報告)"
        )
    old_result = json.loads(old_path.read_text(encoding="utf-8"))

    npz_dir = _persist.NPZ_DIRS[tag]
    chunk_cache: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}
    score_rows_by_sheet = {
        row["sheet_id"]: row
        for row in json.loads((_persist.SCORING_DIR / f"score_{tag}.json").read_text(encoding="utf-8"))
    }

    rechecked: list[dict[str, Any]] = []
    for entry in old_result["persistent_cells"]:
        sheet_row = score_rows_by_sheet.get(entry["sheet_id"])
        if sheet_row is None or sheet_row.get("npz") is None or sheet_row.get("frame_idx") is None:
            # 対応する採点行が見つからない (通常発生しないはず、fail-silent警戒で明記)
            rechecked.append({**entry, "truth_recheck_error": "sheet_row_or_npz_missing"})
            continue
        npz_name = sheet_row["npz"]
        if npz_name not in chunk_cache:
            chunk_cache[npz_name] = _load_chunk_series_with_clock(npz_dir / npz_name)
        series = chunk_cache[npz_name]
        key = (entry["video_id"], entry["side"])
        chunk_rows = series.get(key)
        if not chunk_rows:
            rechecked.append({**entry, "truth_recheck_error": "chunk_rows_missing"})
            continue
        anchor_idx = next(
            (i for i, row in enumerate(chunk_rows) if row["frame_idx"] == sheet_row["frame_idx"]), None,
        )
        if anchor_idx is None:
            rechecked.append({**entry, "truth_recheck_error": "anchor_frame_not_found"})
            continue
        rechecked.append(_recheck_entry(chunk_rows, anchor_idx, entry["r"], entry["c"], entry))

    still_persistent = [e for e in rechecked if not e.get("reclassified_non_persistent", False)]
    reclassified = [e for e in rechecked if e.get("reclassified_non_persistent", False)]
    n_high_confidence = sum(
        1 for e in reclassified if e.get("confidence_tier") == "high_confidence_artifact"
    )
    n_ambiguous = sum(1 for e in reclassified if e.get("confidence_tier") == "ambiguous")

    return {
        "tag": tag,
        "n_persistent_original": old_result["n_persistent"],
        "n_persistent_truth_verified": len(still_persistent),
        "n_reclassified_non_persistent": len(reclassified),
        "n_reclassified_high_confidence_artifact": n_high_confidence,
        "n_reclassified_ambiguous_needs_review": n_ambiguous,
        "cells": rechecked,
        "still_persistent_cells": still_persistent,
        "reclassified_non_persistent_cells": reclassified,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=["a", "b", "c", "d", "e", "f"])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    tags = ["a", "b", "c", "d", "e", "f"] if args.all else ([args.tag] if args.tag else [])
    if not tags:
        ap.print_help()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        result = recheck_tag(tag)
        out_path = OUT_DIR / f"persistent_misread_{tag}_truth_rechecked.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[{tag}] 持続誤認 旧={result['n_persistent_original']} "
            f"新(truth-verified)={result['n_persistent_truth_verified']} "
            f"除外={result['n_reclassified_non_persistent']} "
            f"(高確信={result['n_reclassified_high_confidence_artifact']} "
            f"要レビュー={result['n_reclassified_ambiguous_needs_review']})"
        )
        for e in result["reclassified_non_persistent_cells"]:
            print(
                f"  除外[{e['confidence_tier']}]: {e['sheet_id']} r{e['r']}c{e['c']} "
                f"旧={e['original_frames_equiv']}f -> 新={e['truth_stable_frames_equiv']}f "
                f"(tsumo遷移{e['n_tsumo_transitions_in_original_run']}回)"
            )
        print(f"[{tag}] -> {out_path}")


if __name__ == "__main__":
    main()
