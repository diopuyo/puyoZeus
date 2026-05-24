"""v97m11 cycle 37 board log の深掘り分析 - puyo→empty fail-silent の実態."""
from __future__ import annotations
import json
from pathlib import Path

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN, Board,
)
from src.chain import ChainSimulator


def to_board(grid):
    if grid is None:
        return None
    b = Board()
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            b.set(r, c, int(grid[r][c]))
    return b


def main():
    p = Path("logs/board_logs/cycle37_v97m11.jsonl")
    entries = []
    with open(p) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    print(f"Loaded {len(entries)} frames")

    # 各 side ごとに STABLE 間 puyo 数の連続推移
    for side in ("p1", "p2"):
        print(f"\n=== {side.upper()} STABLE 間 puyo 数推移 ===")
        prev_count = None
        prev_fi = None
        all_drops = []
        for e in entries:
            state = e[f"{side}_state"]
            grid = e.get(f"{side}_confirmed")
            if state != "stable" or grid is None:
                continue
            board = to_board(grid)
            count = board.count_puyos()
            if prev_count is not None:
                diff = count - prev_count
                if diff <= -1:  # 全 drop event を収集
                    all_drops.append({
                        "frame": e["frame_idx"], "t": e["t_sec"],
                        "prev_count": prev_count, "cur_count": count,
                        "diff": diff,
                    })
            prev_count = count
            prev_fi = e["frame_idx"]
        # サマリ
        drop_buckets = {
            "-1": 0, "-2,-3": 0, "-4,-5": 0, "-6,-10": 0, "-11+": 0,
        }
        for d in all_drops:
            v = d["diff"]
            if v == -1:
                drop_buckets["-1"] += 1
            elif -3 <= v <= -2:
                drop_buckets["-2,-3"] += 1
            elif -5 <= v <= -4:
                drop_buckets["-4,-5"] += 1
            elif -10 <= v <= -6:
                drop_buckets["-6,-10"] += 1
            else:
                drop_buckets["-11+"] += 1
        print(f"  drop event 合計: {len(all_drops)} 件")
        for k, v in drop_buckets.items():
            print(f"    diff {k}: {v} 件")
        # 大きな drop の詳細
        big_drops = [d for d in all_drops if d["diff"] <= -3]
        if big_drops:
            print(f"  diff <= -3 詳細 ({len(big_drops)} 件):")
            for d in big_drops[:20]:
                print(f"    frame={d['frame']:>5} t={d['t']:6.2f}s "
                      f"{d['prev_count']}→{d['cur_count']} ({d['diff']:+d})")

    # ツモ数推定 vs confirmed puyo 数
    # 試合開始から終了まで、 STABLE で観測された最大 puyo 数 - 連鎖消滅
    print("\n=== STABLE puyo 数の時系列 (= 全 frame、 5 秒ごと) ===")
    for side in ("p1", "p2"):
        last_at_5s = {}
        for e in entries:
            state = e[f"{side}_state"]
            grid = e.get(f"{side}_confirmed")
            if state != "stable" or grid is None:
                continue
            sec_bucket = int(e["t_sec"] // 5) * 5
            board = to_board(grid)
            last_at_5s[sec_bucket] = board.count_puyos()
        print(f"  {side.upper()}: ", end="")
        for sec in sorted(last_at_5s.keys()):
            print(f"{sec}s={last_at_5s[sec]} ", end="")
        print()


if __name__ == "__main__":
    main()
