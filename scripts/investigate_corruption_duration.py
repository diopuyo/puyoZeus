"""
corruption 持続長分布 診断スクリプト

STABLE 確定盤面の corruption (raw_cnn==raw_hsv=X(非UNKNOWN) かつ confirmed!=X) について:
1. 同一セルで「confirmed=Y のまま CNN==HSV=X が継続」する連続 frame 数の分布
2. recovery gate が最終的に追従するか/ずっと Y のままか
3. color_to_color vs color_to_empty vs empty_to_color の持続長傾向差
4. 長期凍結セルの end_reason 内訳 (state_change / resolved / eof)

実行例:
    python -m scripts.investigate_corruption_duration
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ #
# 定数
# ------------------------------------------------------------------ #
COLOR_UNKNOWN: int = 10
COLOR_EMPTY: int = 0
BOARD_ROWS: int = 13
BOARD_COLS: int = 6

# 対象 board_log ファイル (診断対象、採用 default 版)
DEFAULT_LOGS: list[str] = [
    "data/verify/viz/v89_match01_D_2026-06-03.jsonl",
    "data/verify/viz/v89_match02_D_2026-06-03.jsonl",
    "data/verify/viz/v70_match02_formulaD_2026-06-02.jsonl",
]


def _classify_corruption(confirmed_v: int, target_v: int) -> str:
    """corruption の方向を分類する."""
    if confirmed_v == COLOR_EMPTY:
        return "empty_to_color"
    if target_v == COLOR_EMPTY:
        return "color_to_empty"
    return "color_to_color"


def analyze_file(path: Path) -> list[dict[str, Any]]:
    """board_log JSONL を解析して corruption run リストを返す."""
    print(f"  解析中: {path.name} ...")

    # セルキー: (side, row, col) -> 現在進行中の run 情報
    active_runs: dict[tuple[str, int, int], dict[str, Any]] = {}
    finished_runs: list[dict[str, Any]] = []

    def close_run(
        key: tuple[str, int, int],
        end_frame: int,
        resolved: bool,
        end_reason: str,
    ) -> None:
        run = active_runs.pop(key)
        run["end_frame"] = end_frame
        run["resolved"] = resolved
        run["end_reason"] = end_reason
        finished_runs.append(run)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            frame_idx: int = row["frame_idx"]
            t_sec: float = row["t_sec"]

            for side in ("p1", "p2"):
                state: str = row.get(f"{side}_state", "")
                is_stable: bool = state == "stable"

                # STABLE 外に遷移 → 進行中 run を全 close (state_change)
                if not is_stable:
                    for key in [k for k in active_runs if k[0] == side]:
                        close_run(key, frame_idx - 1, False, "state_change")
                    continue

                cnn_board = row.get(f"{side}_raw_cnn_board")
                hsv_board = row.get(f"{side}_raw_hsv_board")
                conf_board = row.get(f"{side}_confirmed")

                if cnn_board is None or hsv_board is None or conf_board is None:
                    for key in [k for k in active_runs if k[0] == side]:
                        close_run(key, frame_idx - 1, False, "state_change")
                    continue

                # 今フレームの corruption セル集合を構築
                current_corr: set[tuple[int, int]] = set()
                corr_vals: dict[tuple[int, int], tuple[int, int]] = {}
                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        c_v: int = cnn_board[r][c]
                        h_v: int = hsv_board[r][c]
                        cf_v: int = conf_board[r][c]
                        if c_v == COLOR_UNKNOWN or h_v == COLOR_UNKNOWN:
                            continue
                        if c_v != h_v:
                            continue
                        if cf_v == c_v:
                            continue
                        current_corr.add((r, c))
                        corr_vals[(r, c)] = (cf_v, c_v)  # (confirmed_v, target_v)

                # active_runs の中で今フレームに corruption がなくなったセルを close (resolved)
                for key in [k for k in active_runs if k[0] == side and (k[1], k[2]) not in current_corr]:
                    close_run(key, frame_idx, True, "resolved")

                # 今フレームの corruption セルを処理
                for (r, c), (cf_v, t_v) in corr_vals.items():
                    key = (side, r, c)
                    if key in active_runs:
                        run = active_runs[key]
                        if run["conf_v"] == cf_v and run["target_v"] == t_v:
                            # 同じ run が継続
                            run["frame_count"] += 1
                            run["last_frame"] = frame_idx
                        else:
                            # 値が変わった = 前 run close して新規開始
                            close_run(key, frame_idx - 1, False, "value_changed")
                            active_runs[key] = _new_run(side, r, c, frame_idx, t_sec, cf_v, t_v)
                    else:
                        active_runs[key] = _new_run(side, r, c, frame_idx, t_sec, cf_v, t_v)

    # 動画末端まで続いた run (eof)
    for key, run in list(active_runs.items()):
        close_run(key, run["last_frame"], False, "eof")

    return finished_runs


def _new_run(
    side: str, r: int, c: int,
    frame_idx: int, t_sec: float,
    cf_v: int, t_v: int,
) -> dict[str, Any]:
    return {
        "side": side, "row": r, "col": c,
        "start_frame": frame_idx, "last_frame": frame_idx, "end_frame": None,
        "conf_v": cf_v, "target_v": t_v,
        "frame_count": 1,
        "resolved": False, "end_reason": None,
        "start_t": t_sec,
        "kind": _classify_corruption(cf_v, t_v),
    }


def print_stats(all_runs: list[dict[str, Any]], long_threshold: int) -> None:
    """全 run の統計を標準出力に出力する."""
    if not all_runs:
        print("corruption run が 0 件です。")
        return

    total = len(all_runs)
    resolved = [r for r in all_runs if r["resolved"]]
    unresolved = [r for r in all_runs if not r["resolved"]]

    print(f"\n{'='*60}")
    print(f"  corruption run 総数: {total}")
    print(f"  追従解決 (resolved):  {len(resolved)} ({100*len(resolved)/total:.1f}%)")
    print(f"  未解決 (unresolved):  {len(unresolved)} ({100*len(unresolved)/total:.1f}%)")

    # 持続長ヒストグラム (全 run)
    print(f"\n--- 持続長分布 (frame 数、全 run) ---")
    bucket_edges = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100, 10**9]
    bucket_labels = ["1", "2", "3", "4-5", "6-8", "9-10", "11-15", "16-20", "21-30", "31-50", "51-100", "101+"]
    counts = [0] * len(bucket_labels)
    for r in all_runs:
        fc = r["frame_count"]
        for i, b in enumerate(bucket_edges):
            if fc <= b:
                counts[i] += 1
                break
    for label, cnt in zip(bucket_labels, counts):
        bar = "#" * (cnt * 40 // max(1, total))
        pct = 100 * cnt / total
        print(f"  {label:>6} fr: {cnt:5d} ({pct:5.1f}%) {bar}")

    short = sum(1 for r in all_runs if r["frame_count"] <= 8)
    long_ = sum(1 for r in all_runs if r["frame_count"] > long_threshold)
    print(f"\n  <=8fr (recovery gate N 以内): {short} ({100*short/total:.1f}%)")
    print(f"  >{long_threshold}fr (長期凍結):               {long_} ({100*long_/total:.1f}%)")

    # 種別 × 持続長
    print(f"\n--- 種別 x 持続長 ---")
    kinds = ["color_to_empty", "color_to_color", "empty_to_color"]
    for kind in kinds:
        k_runs = [r for r in all_runs if r["kind"] == kind]
        if not k_runs:
            continue
        avg_fc = sum(r["frame_count"] for r in k_runs) / len(k_runs)
        max_fc = max(r["frame_count"] for r in k_runs)
        long_k = sum(1 for r in k_runs if r["frame_count"] > long_threshold)
        res_k = sum(1 for r in k_runs if r["resolved"])
        print(
            f"  {kind}: n={len(k_runs)}, avg={avg_fc:.1f}fr, max={max_fc}fr, "
            f"long(>{long_threshold}fr)={long_k} ({100*long_k/len(k_runs):.0f}%), "
            f"resolved={res_k}/{len(k_runs)} ({100*res_k/len(k_runs):.0f}%)"
        )

    # 長期凍結の詳細
    long_runs = sorted(
        [r for r in all_runs if r["frame_count"] > long_threshold],
        key=lambda x: -x["frame_count"],
    )
    if long_runs:
        print(f"\n--- 長期凍結 (>{long_threshold}fr) サンプル (最大20件) ---")
        for r in long_runs[:20]:
            print(
                f"  side={r['side']} r={r['row']} c={r['col']} "
                f"kind={r['kind']} fc={r['frame_count']} "
                f"conf={r['conf_v']}->target={r['target_v']} "
                f"start={r['start_t']:.1f}s "
                f"resolved={r['resolved']} reason={r['end_reason']}"
            )

    # resolved の解決フレーム数
    if resolved:
        resolve_fcs = [r["frame_count"] for r in resolved]
        avg_r = sum(resolve_fcs) / len(resolve_fcs)
        max_r = max(resolve_fcs)
        within_8 = sum(1 for fc in resolve_fcs if fc <= 8)
        within_15 = sum(1 for fc in resolve_fcs if fc <= 15)
        print(f"\n--- 解決フレーム数 (resolved n={len(resolved)}) ---")
        print(f"  avg={avg_r:.1f}fr, max={max_r}fr")
        print(f"  8fr以内 (=recovery gate N 以内): {within_8}/{len(resolved)} ({100*within_8/len(resolved):.1f}%)")
        print(f"  15fr以内: {within_15}/{len(resolved)} ({100*within_15/len(resolved):.1f}%)")

    # 長期未解決の end_reason 内訳
    long_unres = [r for r in unresolved if r["frame_count"] > long_threshold]
    if long_unres:
        print(f"\n--- 長期未解決 (>{long_threshold}fr, unresolved) n={len(long_unres)} ---")
        reason_count: dict[str, int] = defaultdict(int)
        for r in long_unres:
            reason_count[r["end_reason"]] += 1
        print("  end_reason 内訳:")
        for reason, cnt in sorted(reason_count.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {cnt}")
        print("  サンプル (最大10件, 長い順):")
        for r in sorted(long_unres, key=lambda x: -x["frame_count"])[:10]:
            print(
                f"    side={r['side']} r={r['row']} c={r['col']} "
                f"kind={r['kind']} fc={r['frame_count']} "
                f"conf={r['conf_v']}->target={r['target_v']} "
                f"start={r['start_t']:.1f}s reason={r['end_reason']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="corruption 持続長分布 診断")
    parser.add_argument(
        "--logs", nargs="+",
        default=DEFAULT_LOGS,
        help="解析対象 board_log JSONL ファイル",
    )
    parser.add_argument(
        "--long-threshold", type=int, default=10,
        help="長期凍結と見なす frame 数 (default 10)",
    )
    args = parser.parse_args()

    all_runs: list[dict[str, Any]] = []
    for log_path_str in args.logs:
        p = Path(log_path_str)
        if not p.exists():
            print(f"  [SKIP] not found: {p}")
            continue
        runs = analyze_file(p)
        print(f"    runs: {len(runs)}")
        all_runs.extend(runs)

    print(f"\n=== 全ファイル合算 ({len(args.logs)} ファイル) ===")
    print_stats(all_runs, args.long_threshold)


if __name__ == "__main__":
    main()
