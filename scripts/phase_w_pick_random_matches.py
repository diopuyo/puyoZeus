"""W5: 全動画の matches.tsv から 60-80 秒の中試合をランダム N 試合ピック。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_pick_random_matches \
        --n 3 --min 60 --max 80 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()


def load_matches_for_video(video_id: int) -> list[dict]:
    """match_winners_v0X.tsv から (video_id, idx, start, end, winner) を読む。"""
    vid_short = f"v{video_id:02d}"
    path = Path(f"data/verify/match_winners_{vid_short}.tsv")
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                rows.append({
                    "video_id": video_id,
                    "idx": int(r["idx"]),
                    "start_sec": float(r["start_sec"]),
                    "end_sec": float(r["end_sec"]),
                    "duration": float(r["end_sec"]) - float(r["start_sec"]),
                    "winner": r["winner"].strip(),
                })
            except (KeyError, ValueError):
                continue
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--min", type=float, default=60.0)
    parser.add_argument("--max", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--video-ids", type=int, nargs="+",
        default=list(range(1, 20)),
    )
    parser.add_argument(
        "--out", default="data/verify/phase_w_results/picked_matches.tsv",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_matches: list[dict] = []
    for v in args.video_ids:
        all_matches.extend(load_matches_for_video(v))
    print(f"loaded {len(all_matches)} matches across {len(args.video_ids)} videos")

    candidates = [
        m for m in all_matches
        if args.min <= m["duration"] <= args.max
        and m["winner"] in ("1P", "2P")
    ]
    print(f"  duration {args.min}-{args.max}s candidates: {len(candidates)}")

    if len(candidates) < args.n:
        print(f"WARN: only {len(candidates)} candidates, returning all")
        picked = candidates
    else:
        picked = rng.sample(candidates, args.n)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("video_id\tmatch_idx\tstart_sec\tend_sec\tduration\twinner\n")
        for m in picked:
            f.write(
                f"{m['video_id']:02d}\t{m['idx']}\t"
                f"{m['start_sec']:.1f}\t{m['end_sec']:.1f}\t"
                f"{m['duration']:.1f}\t{m['winner']}\n"
            )

    print(f"\npicked {len(picked)} matches:")
    for m in picked:
        print(
            f"  v{m['video_id']:02d} m{m['idx']:02d}: "
            f"{m['start_sec']:.0f}-{m['end_sec']:.0f}s "
            f"({m['duration']:.0f}s, winner={m['winner']})"
        )
    print(f"\nsaved: {to_windows_path(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
