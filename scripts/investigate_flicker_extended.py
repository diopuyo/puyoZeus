"""ちらつき区間の35fウィンドウ拡張測定 (read-only)"""
import json
import statistics
from pathlib import Path

PROJ = Path(__file__).parent.parent

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

ROWS, COLS = 13, 6
FLICKER_WINDOW = 35

targets = [
    ("data/verify/viz/v89_match01_caseX_2026-06-05.jsonl", "1"),
    ("data/verify/viz/v89_match01_caseX_2026-06-05.jsonl", "2"),
    ("data/verify/viz/v70_match02_caseX_2026-06-05.jsonl", "1"),
    ("data/verify/viz/v70_match02_caseX_2026-06-05.jsonl", "2"),
]

for fname, side in targets:
    frames = load_jsonl(PROJ / fname)
    cnn_key = "p" + side + "_raw_cnn_board"
    state_key = "p" + side + "_state"

    transitions = []
    for i in range(1, len(frames)):
        if frames[i-1].get(state_key) == "chain" and frames[i].get(state_key) == "stable":
            transitions.append(i)

    print(f"\n{fname} side={side}: 遷移{len(transitions)}回 ({FLICKER_WINDOW}f window)")
    all_flicker = []
    for t_idx in transitions:
        t_sec = frames[t_idx].get("t_sec", "?")
        window_frames = []
        for fi in range(t_idx, min(t_idx + FLICKER_WINDOW, len(frames))):
            b = frames[fi].get(cnn_key)
            if b is not None:
                window_frames.append(b)

        max_last_change = 0
        for r in range(ROWS):
            for c in range(COLS):
                last_change = 0
                for fi in range(1, len(window_frames)):
                    if window_frames[fi][r][c] != window_frames[fi-1][r][c]:
                        last_change = fi
                if last_change > max_last_change:
                    max_last_change = last_change

        all_flicker.append(max_last_change)
        next_state_idx = min(t_idx + max_last_change, len(frames) - 1)
        next_state = frames[next_state_idx].get(state_key, "?")
        print(f"  t={t_sec:.2f}s: 最後変化={max_last_change}f  state@change={next_state}")

    print(f"  中央値={statistics.median(all_flicker):.1f}  最大={max(all_flicker)}  分布={sorted(all_flicker)}")
