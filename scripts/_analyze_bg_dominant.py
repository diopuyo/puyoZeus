"""bg_color_dominant 違反の詳細分析: いつ何色が dominant 化したか."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


COLOR_NAMES = {1: "red", 2: "blue", 3: "green", 4: "yellow", 5: "purple", 9: "ojama"}


def main():
    import sys
    target = Path(
        sys.argv[1] if len(sys.argv) > 1
        else "data/verify/retrospective_eval/baseline_v89m3.json"
    )
    if not target.is_file():
        print(f"missing: {target}")
        return
    with open(target, encoding="utf-8") as f:
        d = json.load(f)

    violations = d.get("violations", [])
    bg_dom = [v for v in violations if v["metric"] == "bg_color_dominant"]
    print(f"Total bg_color_dominant violations: {len(bg_dom)}")

    # 色別集計
    by_color = defaultdict(list)
    by_side = defaultdict(int)
    by_color_side = defaultdict(int)
    for v in bg_dom:
        color = v["extra"]["color"]
        side = v["side"]
        by_color[color].append(v)
        by_side[side] += 1
        by_color_side[(side, color)] += 1

    print("\n=== Color 別集計 ===")
    for color, vs in sorted(by_color.items(), key=lambda x: -len(x[1])):
        name = COLOR_NAMES.get(color, str(color))
        ratios = [v["extra"]["ratio"] for v in vs]
        print(
            f"  color={color} ({name}): {len(vs)} 件、 "
            f"ratio 範囲 {min(ratios):.1%}-{max(ratios):.1%}、 "
            f"中央値 {sorted(ratios)[len(ratios)//2]:.1%}"
        )

    print("\n=== Side 別集計 ===")
    for side, count in sorted(by_side.items()):
        print(f"  {side}: {count} 件")

    print("\n=== Side × Color 別集計 ===")
    for (side, color), count in sorted(
        by_color_side.items(), key=lambda x: -x[1],
    ):
        name = COLOR_NAMES.get(color, str(color))
        print(f"  {side} × {name}: {count} 件")

    # 時系列
    print("\n=== 時系列分布 (= 秒ごとの違反数) ===")
    seconds_bucket = defaultdict(int)
    for v in bg_dom:
        sec = int(v["t_sec"] // 10) * 10
        seconds_bucket[sec] += 1
    for sec in sorted(seconds_bucket.keys()):
        bar = "#" * min(seconds_bucket[sec], 50)
        print(f"  {sec:>3}-{sec+10}s : {seconds_bucket[sec]:>3} {bar}")

    # 連続性 = 連続 frame で同じ違反
    print("\n=== 連続違反 (= 連続 frame で同 side × color の bg_dominant) ===")
    sorted_v = sorted(bg_dom, key=lambda v: v["frame_idx"])
    runs = []
    cur_start = None
    cur_key = None
    cur_count = 0
    for v in sorted_v:
        key = (v["side"], v["extra"]["color"])
        if key == cur_key:
            cur_count += 1
        else:
            if cur_count >= 3:
                runs.append((cur_start, cur_count, cur_key))
            cur_start = v["frame_idx"]
            cur_key = key
            cur_count = 1
    if cur_count >= 3:
        runs.append((cur_start, cur_count, cur_key))
    print(f"  連続 3 frame 以上 の違反 run: {len(runs)} 件")
    for start, count, (side, color) in runs[:15]:
        name = COLOR_NAMES.get(color, str(color))
        print(f"    start_frame={start}, {side} × {name}, run_length={count}")


if __name__ == "__main__":
    main()
