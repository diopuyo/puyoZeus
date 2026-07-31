"""反復5 Step1 診断 (2026-07-23): 連鎖開始時点の「起点盤面」取得可否を数値確認する。

修正なし・計測のみ。scripts/recognition_physics_review.py の capture 基盤を
再利用し、各連鎖トリガー (chain_event.before_board) について:
  1. 全 EMPTY か (捕捉失敗の強い兆候)
  2. 直前の有効 STABLE confirmed_board との cell 差分が大きすぎないか
     (通常は 1 手 = 2 cell 差分程度のはず。 大きければ起点が信頼できない)
を判定し、「起点盤面がそのまま使えるか」の内訳を報告する。

使い方:
    PYTHONPATH=. python -m scripts._diag_step1_startboard_availability
"""
from __future__ import annotations

import numpy as np

from scripts.recognition_physics_review import (
    TARGET_WINDOWS, _capture_frames, _new_chain_triggers, _FrameRecord,
)
from src.board import COLOR_EMPTY

# 直前 STABLE との乖離がこれを超えたら「起点盤面が信頼できない」とみなす
# (cycle 48 の大量 hallucination ガード基準 6 cell を流用)。
DIFF_SUSPECT_CELL_THRESHOLD: int = 6


def _find_last_valid_stable_before(
    records: list[_FrameRecord], idx: int,
) -> _FrameRecord | None:
    """idx より前で最後に state==STABLE かつ grid 有効だったレコードを返す。"""
    for i in range(idx - 1, -1, -1):
        if records[i].state == "STABLE" and records[i].grid is not None:
            return records[i]
    return None


def _analyze_video(stem: str, start_sec: float, max_sec: float) -> dict:
    """1 動画分の起点盤面取得可否を集計する。"""
    by_side = _capture_frames(stem, start_sec, max_sec)
    total = 0
    empty_before = 0
    no_prior_stable = 0
    large_diff = 0
    diffs: list[int] = []
    for side in ("1P", "2P"):
        records = by_side[side]
        for idx in _new_chain_triggers(records):
            rec = records[idx]
            if rec.chain_before_grid is None:
                continue
            total += 1
            puyo_count = int((rec.chain_before_grid != COLOR_EMPTY).sum())
            if puyo_count == 0:
                empty_before += 1
                continue
            prior = _find_last_valid_stable_before(records, idx)
            if prior is None:
                no_prior_stable += 1
                continue
            diff = int((rec.chain_before_grid != prior.grid).sum())
            diffs.append(diff)
            if diff > DIFF_SUSPECT_CELL_THRESHOLD:
                large_diff += 1
    return {
        "video_stem": stem, "total": total, "empty_before": empty_before,
        "no_prior_stable": no_prior_stable, "large_diff": large_diff,
        "diffs": diffs,
    }


def main() -> None:
    """対象動画すべてで起点盤面取得可否を測定し、内訳を出力する。"""
    all_results = []
    for stem, start_sec, max_sec in TARGET_WINDOWS:
        print(f"  {stem}: start={start_sec}s max={max_sec}s を処理中...")
        r = _analyze_video(stem, start_sec, max_sec)
        all_results.append(r)
        print(
            f"    total={r['total']} empty_before={r['empty_before']} "
            f"no_prior_stable={r['no_prior_stable']} "
            f"large_diff(>{DIFF_SUSPECT_CELL_THRESHOLD})={r['large_diff']}",
        )

    total = sum(r["total"] for r in all_results)
    empty_before = sum(r["empty_before"] for r in all_results)
    no_prior_stable = sum(r["no_prior_stable"] for r in all_results)
    large_diff = sum(r["large_diff"] for r in all_results)
    all_diffs = [d for r in all_results for d in r["diffs"]]
    print("\n==== Step1 起点盤面 取得可否 サマリ ====")
    print(f"対象連鎖トリガー数: {total}")
    if total > 0:
        print(
            f"before_board が全 EMPTY (捕捉失敗疑い): {empty_before} "
            f"({empty_before / total:.3f})",
        )
        print(
            f"直前の有効 STABLE が見つからない: {no_prior_stable} "
            f"({no_prior_stable / total:.3f})",
        )
        usable_base = total - empty_before - no_prior_stable
        if usable_base > 0:
            print(
                f"直前STABLEとの diff > {DIFF_SUSPECT_CELL_THRESHOLD}cell "
                f"(乖離大・信頼度低): {large_diff} ({large_diff / usable_base:.3f} "
                f"/ 比較可能な{usable_base}件中)",
            )
    if all_diffs:
        print(
            f"diff cell数: 平均={np.mean(all_diffs):.2f} "
            f"中央値={np.median(all_diffs):.1f} 最大={np.max(all_diffs)} "
            f"件数={len(all_diffs)}",
        )
    usable = total - empty_before - no_prior_stable - large_diff
    print(
        f"\n=> 起点盤面がそのまま使える (empty でなく直前STABLEと整合): "
        f"{usable}/{total} ({usable / total:.3f})" if total > 0 else "対象なし",
    )


if __name__ == "__main__":
    main()
