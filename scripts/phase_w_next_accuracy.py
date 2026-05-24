"""W11: NextDetector の精度を P1/P2 一致率で間接測定。

ぷよぷよでは同じツモが両側に表示される (上下入れ替わるが順序は保たれる)。
1P と 2P の next_pair が一致するべき (色集合として)。一致しないなら NextDetector の誤認。
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console  # noqa: E402
init_console()

import cv2

from src.next_detector import NextDetector


def main() -> int:
    det = NextDetector.load_default()
    results = []
    for vid in ["01", "02", "03", "04", "05", "06", "10", "15", "19"]:
        video_path = f"data/frames/video_{vid}.mp4"
        winners_path = f"data/verify/match_winners_v{vid}.tsv"
        if not Path(winners_path).exists():
            continue
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        with open(winners_path) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        # 最初の 5 試合 × 4 時刻
        for m in rows[:5]:
            s = float(m["start_sec"])
            e = float(m["end_sec"])
            for ratio in [0.2, 0.4, 0.6, 0.8]:
                t = s + (e - s) * ratio
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, fr = cap.read()
                if not ok:
                    continue
                if fr.shape[:2] != (1080, 1920):
                    fr = cv2.resize(
                        fr, (1920, 1080), interpolation=cv2.INTER_AREA,
                    )
                r = det.detect_both(fr)
                p1n = sorted(r.p1.next_pair)
                p2n = sorted(r.p2.next_pair)
                p1d = sorted(r.p1.dnext_pair)
                p2d = sorted(r.p2.dnext_pair)
                next_match = p1n == p2n
                dnext_match = p1d == p2d
                results.append({
                    "video": vid, "t": t,
                    "p1_next": tuple(p1n), "p2_next": tuple(p2n),
                    "p1_dnext": tuple(p1d), "p2_dnext": tuple(p2d),
                    "next_match": next_match,
                    "dnext_match": dnext_match,
                })
        cap.release()

    n = len(results)
    next_n = sum(1 for r in results if r["next_match"])
    dnext_n = sum(1 for r in results if r["dnext_match"])
    print(f"samples: {n}")
    print(f"  next P1/P2 agree:  {next_n}/{n} = {next_n / max(1, n):.2%}")
    print(f"  dnext P1/P2 agree: {dnext_n}/{n} = {dnext_n / max(1, n):.2%}")

    print("\nnext disagreements (first 15):")
    cnt = 0
    for r in results:
        if r["next_match"]:
            continue
        v = r["video"]
        t = r["t"]
        p1 = r["p1_next"]
        p2 = r["p2_next"]
        print(f"  v{v} t={t:.0f}: p1={p1} p2={p2}")
        cnt += 1
        if cnt >= 15:
            break

    # 動画別集計
    print("\nper-video next agreement:")
    by_vid: dict[str, list] = {}
    for r in results:
        by_vid.setdefault(r["video"], []).append(r)
    for v, rs in by_vid.items():
        agree = sum(1 for r in rs if r["next_match"])
        print(f"  v{v}: {agree}/{len(rs)} = {agree / len(rs):.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
