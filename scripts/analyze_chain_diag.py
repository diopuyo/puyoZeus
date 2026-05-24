"""diagnose_chain_transitions.py の JSONL を読んで仮説 A/B を判別する.

判別アルゴリズム:
    1. chain_event が None → True に切り替わった frame (= 連鎖検出 frame) を抽出
    2. その frame の sim_chain_count_confirmed_now を見る:
       - >= 1 (= confirmed_board で連鎖発生可能) → confirmed は概ね正しい
                                                  → 仮説 A は否定的
       - == 0 (= confirmed_board では連鎖発生不可) → confirmed の色が違う
                                                  → 仮説 A 真寄り
    3. その frame の前 30 frame で:
       - 初めて score_delta > 0 を観測した frame (= 実際に連鎖が画面で起きた frame)
       - その frame と chain_event 検出 frame の差分 = chain detection 遅延 (frame)
       - 遅延中の confirmed_diff_to_prev の合計 = 動画化アニメ色が confirmed
         board に混入した cell 数
    4. summary を Markdown で出力.

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.analyze_chain_diag \\
        --input data/diagnostics/v91_match1_75s_diag.jsonl \\
        --output data/diagnostics/v91_match1_75s_diag_summary.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_chain_events(
    rows: list[dict[str, Any]], side: str,
) -> list[dict[str, Any]]:
    """side ('p1'/'p2') の chain_event 検出 frame (前 None → 今 not None) を抽出."""
    events: list[dict[str, Any]] = []
    prev_ev: dict[str, Any] | None = None
    for row in rows:
        side_data = row[side]
        ev = side_data.get("chain_event")
        if ev is not None and (
            prev_ev is None
            or prev_ev.get("trigger_sec") != ev.get("trigger_sec")
        ):
            events.append({
                "frame_idx": row["frame_idx"],
                "time_sec": row["time_sec"],
                "state": side_data["state"],
                "chain_count": ev["chain_count"],
                "trigger_sec": ev["trigger_sec"],
                "before_count": ev.get("before_count", 0),
                "sim_chain_count_confirmed_now": ev.get(
                    "sim_chain_count_confirmed_now", 0,
                ),
                "confirmed_total_now": side_data["confirmed_total"],
                "confirmed_grid_hash_now": side_data["confirmed_grid_hash"],
                "cnn_total_now": side_data["cnn_total"],
            })
        prev_ev = ev
    return events


def analyze_pre_event_window(
    rows: list[dict[str, Any]],
    event_frame: int,
    side: str,
    pre_window: int = 60,  # 1 秒分 (60 fps)
) -> dict[str, Any]:
    """連鎖発火 frame の直前 N frame の動きを分析."""
    pre_rows = [
        r for r in rows
        if event_frame - pre_window <= r["frame_idx"] <= event_frame
    ]
    if not pre_rows:
        return {}
    first_score_delta_frame: int | None = None
    sum_confirmed_diffs = 0
    n_chain_state_frames = 0
    state_seq: list[str] = []
    for r in pre_rows:
        side_data = r[side]
        if (
            first_score_delta_frame is None
            and side_data["score_delta"] > 0
            and r["frame_idx"] < event_frame  # 連鎖発火 frame 自身は除く
        ):
            first_score_delta_frame = r["frame_idx"]
        sum_confirmed_diffs += max(0, side_data.get(
            "confirmed_diff_to_prev", 0,
        ))
        if side_data["state"] == "chain":
            n_chain_state_frames += 1
        state_seq.append(f"{r['frame_idx']}:{side_data['state']}")
    return {
        "first_score_delta_frame": first_score_delta_frame,
        "score_delta_to_event_lag": (
            event_frame - first_score_delta_frame
            if first_score_delta_frame is not None else None
        ),
        "sum_confirmed_diffs_in_window": sum_confirmed_diffs,
        "chain_state_frames_in_window": n_chain_state_frames,
    }


def classify_hypothesis(
    event: dict[str, Any], pre_analysis: dict[str, Any],
) -> str:
    """仮説 A/B/両方 を判別 (cycle 71 Phase 1a 対応).

    Phase 1a (= 物理推論主軸化) 後は、 着地時に既に ChainSimulator が連鎖を
    先回り計算し、 confirmed_board が「連鎖後 final」 に書き換わる. その結果:
    - VideoChainTracker 経由で遅延受信された chain_event の時点では、
      confirmed は既に連鎖後 final → puyo 大幅減 → ChainSimulator(confirmed) は 0
    - 旧評価軸ではこれが A=hit と誤判定された

    新評価軸:
    - A=preempt: confirmed_total が before_count より大幅減 (= 連鎖先回り完了)
                 = Phase 1a の正しい挙動、 仮説 A 良判定
    - A=clean: confirmed で連鎖再現可能 (= 旧経路で連鎖前盤面が正しい)
    - A=hit: 連鎖再現不可かつ puyo 減も無い (= 真の置き誤認)
    - A=partial: 部分的に連鎖再現可能
    """
    sim_now = event["sim_chain_count_confirmed_now"]
    lag = pre_analysis.get("score_delta_to_event_lag")
    confirmed_drift = pre_analysis.get("sum_confirmed_diffs_in_window", 0)
    confirmed_now = event["confirmed_total_now"]
    before_count = event["before_count"]
    chain_count = event["chain_count"]
    flags: list[str] = []
    # Phase 1a preempt 判定:
    # 連鎖発生時、 物理推論で chain_count 連鎖分の puyo が消えた状態が confirmed.
    # 大雑把に「連鎖 1 段 = 4 puyo 消去」 とすると、 confirmed が
    # before_count - 4 × chain_count 以下なら preempt 完了とみなす.
    expected_erase = max(4, 4 * chain_count)
    if (
        before_count > 0
        and confirmed_now <= before_count - expected_erase + 2
        # +2 は余裕 (= 厳密一致でなくても近ければ OK)
    ):
        flags.append("A=preempt")
    elif sim_now == 0 and confirmed_now > 4:
        flags.append("A=hit")
    elif sim_now >= chain_count:
        flags.append("A=clean")
    else:
        flags.append("A=partial")
    # 仮説 B: score_delta 観測から chain_event 受信までの遅延が大きい.
    # 1 frame = 1/60 sec. 5 frame 以上の遅延を suspicious と扱う.
    if lag is None:
        flags.append("B=no_score_signal")
    elif lag >= 5:
        flags.append(f"B=hit(lag={lag}f)")
    else:
        flags.append(f"B=clean(lag={lag}f)")
    # 遅延中の confirmed_diff 大 → animation 色混入 or preempt 反映の可能性
    if confirmed_drift >= 4:
        flags.append(f"drift={confirmed_drift}")
    return " | ".join(flags)


def summarize(
    rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# 仮説 A/B 判別 summary")
    lines.append("")
    lines.append(f"- 総 frame 数: {len(rows)}")
    if rows:
        lines.append(
            f"- 時刻範囲: {rows[0]['time_sec']:.2f}s 〜 {rows[-1]['time_sec']:.2f}s",
        )
    lines.append("")
    # state distribution
    state_counts: dict[str, dict[str, int]] = {
        "p1": defaultdict(int),
        "p2": defaultdict(int),
    }
    for r in rows:
        state_counts["p1"][r["p1"]["state"]] += 1
        state_counts["p2"][r["p2"]["state"]] += 1
    lines.append("## state 分布")
    for side in ["p1", "p2"]:
        lines.append(f"### {side}")
        for st, n in sorted(
            state_counts[side].items(), key=lambda x: -x[1],
        ):
            pct = n * 100 / max(1, len(rows))
            lines.append(f"- {st}: {n} ({pct:.1f}%)")
        lines.append("")
    # chain events
    for side in ["p1", "p2"]:
        events = find_chain_events(rows, side)
        lines.append(f"## {side} chain events: {len(events)} 件")
        lines.append("")
        if not events:
            lines.append("(検出 0)")
            lines.append("")
            continue
        lines.append(
            "| ev_idx | frame | t_sec | state | n連鎖 | "
            "trigger_sec | before_cnt | confirmed_now | "
            "sim_chain(confirmed) | 仮説判定 |",
        )
        lines.append(
            "|---|---|---|---|---|---|---|---|---|---|",
        )
        for i, ev in enumerate(events):
            pre = analyze_pre_event_window(rows, ev["frame_idx"], side)
            verdict = classify_hypothesis(ev, pre)
            lines.append(
                f"| {i} | {ev['frame_idx']} | {ev['time_sec']:.2f} | "
                f"{ev['state']} | {ev['chain_count']} | "
                f"{ev['trigger_sec']:.2f} | {ev['before_count']} | "
                f"{ev['confirmed_total_now']} | "
                f"{ev['sim_chain_count_confirmed_now']} | "
                f"{verdict} |",
            )
        lines.append("")
        # 詳細
        lines.append(f"### {side} 詳細")
        lines.append("")
        for i, ev in enumerate(events):
            pre = analyze_pre_event_window(rows, ev["frame_idx"], side)
            lines.append(
                f"- ev[{i}] frame={ev['frame_idx']} "
                f"t={ev['time_sec']:.2f}s "
                f"n連鎖={ev['chain_count']} "
                f"trigger_sec={ev['trigger_sec']:.2f}",
            )
            lines.append(f"  - before_count={ev['before_count']}")
            lines.append(
                f"  - confirmed_total_now={ev['confirmed_total_now']} "
                f"hash={ev['confirmed_grid_hash_now']}",
            )
            lines.append(
                f"  - sim_chain_count_confirmed_now="
                f"{ev['sim_chain_count_confirmed_now']}",
            )
            if pre:
                lines.append(
                    f"  - first_score_delta_frame="
                    f"{pre.get('first_score_delta_frame')}",
                )
                lines.append(
                    f"  - score_delta_to_event_lag="
                    f"{pre.get('score_delta_to_event_lag')} frame",
                )
                lines.append(
                    f"  - sum_confirmed_diffs_in_window="
                    f"{pre.get('sum_confirmed_diffs_in_window')}",
                )
                lines.append(
                    f"  - chain_state_frames_in_window="
                    f"{pre.get('chain_state_frames_in_window')}",
                )
        lines.append("")
    # 仮説 A/B 集計
    lines.append("## 仮説 A/B 集計")
    counter_a: dict[str, int] = defaultdict(int)
    counter_b: dict[str, int] = defaultdict(int)
    for side in ["p1", "p2"]:
        for ev in find_chain_events(rows, side):
            pre = analyze_pre_event_window(rows, ev["frame_idx"], side)
            verdict = classify_hypothesis(ev, pre)
            for part in verdict.split(" | "):
                if part.startswith("A="):
                    counter_a[part] += 1
                elif part.startswith("B="):
                    counter_b[part] += 1
    lines.append("### 仮説 A (置き誤認 → freeze)")
    for k, n in sorted(counter_a.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {n}")
    lines.append("")
    lines.append("### 仮説 B (chain detection 遅延)")
    for k, n in sorted(counter_b.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {n}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    summarize(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
