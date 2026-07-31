"""color_to_color の recovery gate 到達率と delayed-consensus 効果推計."""
import json
from pathlib import Path

COLOR_UNKNOWN = 10
COLOR_EMPTY = 0
BOARD_ROWS = 13
BOARD_COLS = 6
STABLE_WARMUP_FRAMES = 12

logs = [
    "data/verify/viz/v89_match01_D_2026-06-03.jsonl",
    "data/verify/viz/v89_match02_D_2026-06-03.jsonl",
    "data/verify/viz/v70_match02_formulaD_2026-06-02.jsonl",
]

all_runs: list[dict] = []

for log_path in logs:
    p = Path(log_path)
    if not p.exists():
        continue

    with open(p) as f:
        lines = f.readlines()
    rows = [json.loads(l) for l in lines]

    active_runs: dict = {}
    finished_runs: list[dict] = []

    def close_run(key: tuple, end_frame: int, resolved: bool, end_reason: str) -> None:
        run = active_runs.pop(key)
        run["end_frame"] = end_frame
        run["resolved"] = resolved
        run["end_reason"] = end_reason
        finished_runs.append(run)

    stable_starts: dict[str, int | None] = {"p1": None, "p2": None}

    for row in rows:
        frame_idx = row["frame_idx"]
        t_sec = row["t_sec"]

        for side in ("p1", "p2"):
            state = row.get(f"{side}_state", "")
            is_stable = state == "stable"

            if not is_stable:
                stable_starts[side] = None
                for key in [k for k in active_runs if k[0] == side]:
                    close_run(key, frame_idx - 1, False, "state_change")
                continue

            if stable_starts[side] is None:
                stable_starts[side] = frame_idx

            elapsed = frame_idx - stable_starts[side]

            cnn_board = row.get(f"{side}_raw_cnn_board")
            hsv_board = row.get(f"{side}_raw_hsv_board")
            conf_board = row.get(f"{side}_confirmed")
            if cnn_board is None or hsv_board is None or conf_board is None:
                continue

            current_corr: set[tuple[int, int]] = set()
            corr_vals: dict[tuple[int, int], tuple] = {}
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    c_v = cnn_board[r][c]
                    h_v = hsv_board[r][c]
                    cf_v = conf_board[r][c]
                    if c_v == COLOR_UNKNOWN or h_v == COLOR_UNKNOWN:
                        continue
                    if c_v != h_v:
                        continue
                    if cf_v == c_v:
                        continue
                    if cf_v == COLOR_EMPTY:
                        kind = "etc"
                    elif c_v == COLOR_EMPTY:
                        kind = "cte"
                    else:
                        kind = "ctc"
                    current_corr.add((r, c))
                    corr_vals[(r, c)] = (cf_v, c_v, kind, elapsed)

            for key in [k for k in active_runs if k[0] == side and (k[1], k[2]) not in current_corr]:
                close_run(key, frame_idx, True, "resolved")

            for (r, c), (cf_v, t_v, kind, el) in corr_vals.items():
                key = (side, r, c)
                if key in active_runs:
                    run = active_runs[key]
                    if run["conf_v"] == cf_v and run["target_v"] == t_v:
                        run["frame_count"] += 1
                        run["last_frame"] = frame_idx
                        run["last_elapsed"] = el
                    else:
                        close_run(key, frame_idx - 1, False, "value_changed")
                        active_runs[key] = {"side": side, "row": r, "col": c, "start_frame": frame_idx, "last_frame": frame_idx, "end_frame": None, "conf_v": cf_v, "target_v": t_v, "frame_count": 1, "resolved": False, "end_reason": None, "start_t": t_sec, "kind": kind, "start_elapsed": el, "last_elapsed": el}
                else:
                    active_runs[key] = {"side": side, "row": r, "col": c, "start_frame": frame_idx, "last_frame": frame_idx, "end_frame": None, "conf_v": cf_v, "target_v": t_v, "frame_count": 1, "resolved": False, "end_reason": None, "start_t": t_sec, "kind": kind, "start_elapsed": el, "last_elapsed": el}

    for key, run in list(active_runs.items()):
        close_run(key, run["last_frame"], False, "eof")

    all_runs.extend(finished_runs)

print("=== 種別 x gate 到達 x resolved ===")
for kind in ("etc", "cte", "ctc"):
    runs = [r for r in all_runs if r["kind"] == kind]
    if not runs:
        continue
    fc_ge8 = [r for r in runs if r["frame_count"] >= 8]
    fc_ge8_resolved = [r for r in fc_ge8 if r["resolved"]]
    fc_lt8 = [r for r in runs if r["frame_count"] < 8]
    fc_lt8_resolved = [r for r in fc_lt8 if r["resolved"]]
    print(f"  {kind}: n={len(runs)}")
    print(f"    fc>=8 (gate閾値到達): {len(fc_ge8)}, うち resolved={len(fc_ge8_resolved)} ({100*len(fc_ge8_resolved)/max(1,len(fc_ge8)):.0f}%)")
    print(f"    fc<8  (gate閾値未達): {len(fc_lt8)}, うち resolved={len(fc_lt8_resolved)} ({100*len(fc_lt8_resolved)/max(1,len(fc_lt8)):.0f}%)")

print()
print("=== color_to_color warmup別 ===")
ctc = [r for r in all_runs if r["kind"] == "ctc"]
in_warmup = [r for r in ctc if r.get("start_elapsed", 0) <= STABLE_WARMUP_FRAMES]
after_warmup = [r for r in ctc if r.get("start_elapsed", 0) > STABLE_WARMUP_FRAMES]
print(f"  warmup内開始: {len(in_warmup)}, warmup後開始: {len(after_warmup)}")
print(f"  warmup内 fc>=8: {sum(1 for r in in_warmup if r['frame_count']>=8)}")
print(f"  warmup後 fc>=8: {sum(1 for r in after_warmup if r['frame_count']>=8)}")

print()
# 理論上 recovery gate が発火すべきなのに発火しない ctc:
# warmup後, fc>=8, unresolved だがend_reason=state_change
ctc_long_unres = [r for r in ctc if r["frame_count"] >= 8 and not r["resolved"]]
print(f"=== color_to_color fc>=8 unresolved: {len(ctc_long_unres)} ===")
for r in ctc_long_unres[:10]:
    print(f"  side={r['side']} r={r['row']} c={r['col']} fc={r['frame_count']} "
          f"start_elapsed={r.get('start_elapsed',0)} reason={r['end_reason']}")
