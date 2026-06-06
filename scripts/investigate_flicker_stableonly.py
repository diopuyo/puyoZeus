"""
chain→stable遷移後の「stable期間中だけ」でのちらつき持続測定 (read-only)

エフェクト残光の純粋な持続時間 = stable期間中にraw_cnnが変化し続ける区間。
次の非stable状態が来た時点で打ち切る。
"""
import json
import statistics
from pathlib import Path

PROJ = Path(__file__).parent.parent

ROWS, COLS = 13, 6


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def measure_stable_only_flicker(frames: list, side: str, label: str) -> None:
    """stable期間中のみでのraw_cnnちらつき持続フレーム数を測定する"""
    cnn_key = f"p{side}_raw_cnn_board"
    state_key = f"p{side}_state"

    transitions = [
        i for i in range(1, len(frames))
        if frames[i-1].get(state_key) == "chain"
        and frames[i].get(state_key) == "stable"
    ]

    print(f"\n=== stable期間限定ちらつき測定 [{label} side={side}] ===")
    print(f"  遷移回数: {len(transitions)}")

    flicker_durations = []

    for t_idx in transitions:
        t_sec = frames[t_idx].get("t_sec", "?")

        # stable期間のみのフレームを収集
        stable_frames_boards = []
        stable_end_offset = 0
        for offset in range(120):  # 最大4秒(30fps)
            fi = t_idx + offset
            if fi >= len(frames):
                break
            if frames[fi].get(state_key) != "stable":
                stable_end_offset = offset
                break
            b = frames[fi].get(cnn_key)
            if b is not None:
                stable_frames_boards.append((offset, b))
        else:
            stable_end_offset = 120

        # stable期間中の最後のraw_cnn変化offset
        max_last_change = 0
        for r in range(ROWS):
            for c in range(COLS):
                last_change = 0
                for idx in range(1, len(stable_frames_boards)):
                    prev_b = stable_frames_boards[idx-1][1]
                    curr_b = stable_frames_boards[idx][1]
                    if curr_b[r][c] != prev_b[r][c]:
                        last_change = stable_frames_boards[idx][0]
                if last_change > max_last_change:
                    max_last_change = last_change

        stable_duration = stable_end_offset if stable_end_offset > 0 else len(stable_frames_boards)
        flicker_durations.append(max_last_change)
        print(
            f"  t={t_sec:.2f}s: "
            f"stable継続={stable_duration}f  "
            f"ちらつき最終={max_last_change}f  "
            f"(stable内{len(stable_frames_boards)}フレーム観測)"
        )

    if flicker_durations:
        print(f"\n  --- 集計 ---")
        print(f"  件数: {len(flicker_durations)}")
        print(f"  中央値: {statistics.median(flicker_durations):.1f} frame")
        print(f"  平均:   {statistics.mean(flicker_durations):.1f} frame")
        print(f"  最大:   {max(flicker_durations)} frame")
        print(f"  分布: {sorted(flicker_durations)}")
        settled = sum(1 for v in flicker_durations if v <= 5)
        print(f"  5f以内に静定: {settled}/{len(flicker_durations)} ({100*settled/len(flicker_durations):.0f}%)")
        settled10 = sum(1 for v in flicker_durations if v <= 10)
        print(f"  10f以内に静定: {settled10}/{len(flicker_durations)} ({100*settled10/len(flicker_durations):.0f}%)")


def main() -> None:
    targets = [
        ("data/verify/viz/v89_match01_caseX_2026-06-05.jsonl", "1"),
        ("data/verify/viz/v89_match01_caseX_2026-06-05.jsonl", "2"),
        ("data/verify/viz/v70_match02_caseX_2026-06-05.jsonl", "1"),
        ("data/verify/viz/v70_match02_caseX_2026-06-05.jsonl", "2"),
    ]
    for fname, side in targets:
        path = PROJ / fname
        label = Path(fname).stem
        frames = load_jsonl(path)
        measure_stable_only_flicker(frames, side, label)


if __name__ == "__main__":
    main()
