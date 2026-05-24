"""A=hit 17 件の per-frame context を抽出して原因を分類する.

各 A=hit frame の前後 ±N frame の state / score / chain_event / confirmed_diff を
表形式で出力し、真因分類 (認識崩壊 / state machine 誤動作 / chain count 過大)
を補助する。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.dump_a_hit_context \\
        --input data/diagnostics/v91_match1_75s_diag_phase1b.jsonl \\
        --output data/diagnostics/v91_match1_75s_a_hit_context.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# (side, frame_idx) のリスト = AB summary から目視抽出した A=hit 17 frame
A_HIT_FRAMES: list[tuple[str, int]] = [
    # p1 cluster 1 (25.18-25.52s, chain=1, drift 74-83)
    ("p1", 1511), ("p1", 1515), ("p1", 1519),
    ("p1", 1523), ("p1", 1527), ("p1", 1531),
    # p1 cluster 2 (57.43-57.63s, chain=5, drift=5)
    ("p1", 3446), ("p1", 3450), ("p1", 3454), ("p1", 3458),
    # p2 single (40.37s, chain=2, drift=113)
    ("p2", 2422),
    # p2 cluster (51.13-51.40s, chain=1, drift 39-82)
    ("p2", 3068), ("p2", 3072), ("p2", 3076),
    ("p2", 3080), ("p2", 3083), ("p2", 3084),
]

WINDOW: int = 5  # 前後 frame 数


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _summarize_side(side_data: dict[str, Any]) -> str:
    state = side_data.get("state", "?")
    cnn_total = side_data.get("cnn_total", 0)
    conf_total = side_data.get("confirmed_total", 0)
    score = side_data.get("score", 0)
    score_delta = side_data.get("score_delta", 0)
    ev = side_data.get("chain_event")
    diff = side_data.get("confirmed_diff_to_prev", 0)
    ev_str = (
        f"ev(n={ev.get('chain_count')},t={ev.get('trigger_sec'):.2f})"
        if ev is not None else "-"
    )
    return (
        f"{state:11s} cnn={cnn_total:3d} conf={conf_total:3d} "
        f"diff={diff:3d} score={score:7d} d={score_delta:+5d} {ev_str}"
    )


def dump_context(
    rows: list[dict[str, Any]],
    target_side: str,
    target_frame: int,
) -> list[str]:
    lines = [f"### {target_side} frame={target_frame}"]
    lines.append("| frame | time | side state | summary |")
    lines.append("|---|---|---|---|")
    for row in rows:
        fi = row["frame_idx"]
        if abs(fi - target_frame) > WINDOW:
            continue
        marker = "**>>>**" if fi == target_frame else ""
        t = row["time_sec"]
        side_data = row[target_side]
        summary = _summarize_side(side_data)
        lines.append(f"| {marker}{fi}{marker} | {t:.2f}s | {target_side} | `{summary}` |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    out: list[str] = ["# A=hit 17 件 per-frame context", ""]
    for side, frame in A_HIT_FRAMES:
        out.extend(dump_context(rows, side, frame))
    args.output.write_text("\n".join(out), encoding="utf-8")
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
