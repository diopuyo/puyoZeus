"""特定 frame 範囲の board 推移を可視化."""
from __future__ import annotations
import json
import sys
from pathlib import Path

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, Board


COLOR_CHAR = {0: ".", 1: "R", 2: "B", 3: "G", 4: "Y", 5: "P", 9: "O", 10: "?"}


def to_board(grid):
    if grid is None:
        return None
    b = Board()
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            b.set(r, c, int(grid[r][c]))
    return b


def render(grid, side_name):
    if grid is None:
        return [f"{side_name}: None"]
    lines = [f"{side_name}:"]
    for r in range(1, BOARD_ROWS):  # 可視 12 行
        row = "  "
        for c in range(BOARD_COLS):
            row += COLOR_CHAR.get(int(grid[r][c]), "?")
        lines.append(row)
    return lines


def main(path, t_start, t_end, side):
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    print(f"Loaded {len(entries)} frames")
    # 範囲内の entry を STABLE 中心に sample
    samples = []
    for e in entries:
        t = e["t_sec"]
        if t < t_start or t > t_end:
            continue
        samples.append(e)
    # 1 秒ごとに 1 frame 抽出
    by_sec = {}
    for s in samples:
        sec = int(s["t_sec"] * 2) / 2  # 0.5 秒刻み
        if sec not in by_sec:
            by_sec[sec] = s
    print(f"\n=== {side} 推移 ({t_start}-{t_end}s、 0.5 秒刻み) ===")
    prefix = "p1" if side == "1P" else "p2"
    for sec in sorted(by_sec.keys()):
        e = by_sec[sec]
        state = e[f"{prefix}_state"]
        grid = e.get(f"{prefix}_confirmed")
        b = to_board(grid)
        non_empty = b.count_puyos() if b else 0
        print(f"\nt={sec:6.2f}s  frame={e['frame_idx']}  state={state}  non_empty={non_empty}")
        if b is not None:
            grid_data = grid
            for r in range(1, BOARD_ROWS):
                row = "  "
                for c in range(BOARD_COLS):
                    row += COLOR_CHAR.get(int(grid_data[r][c]), "?")
                print(row)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
