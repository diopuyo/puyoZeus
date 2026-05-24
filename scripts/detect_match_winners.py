"""matches.tsv から各試合の勝者を判定して TSV 出力。"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2

from src.match_winner import MatchWinnerDetector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--matches-tsv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--last-observable-offset", type=float, default=10.0,
                        help="最終試合終了からこの秒数後の数値画像を比較対象に使う")
    args = parser.parse_args()

    detector = MatchWinnerDetector.load_default()
    cap = cv2.VideoCapture(args.video)

    rows: list[dict] = []
    with open(args.matches_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    print(f"試合数: {len(rows)}")

    match_starts = [float(r["start_sec"]) for r in rows]
    last_end = float(rows[-1]["end_sec"]) if rows else 0.0
    last_observable = last_end + args.last_observable_offset

    results = detector.detect_all_winners(
        cap=cap,
        match_starts=match_starts,
        last_observable_sec=last_observable,
        offset_before=1.0,
    )
    cap.release()

    # 出力
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("idx\tstart_sec\tend_sec\twinner\tleft_hamming\tright_hamming\tleft_changed\tright_changed\n")
        for r, w in zip(rows, results):
            f.write(
                f"{r['idx']}\t{r['start_sec']}\t{r['end_sec']}\t"
                f"{w.winner or 'UNKNOWN'}\t"
                f"{w.left_hamming}\t{w.right_hamming}\t"
                f"{int(w.left_changed)}\t{int(w.right_changed)}\n"
            )

    # サマリ
    n_1p = sum(1 for w in results if w.winner == "1P")
    n_2p = sum(1 for w in results if w.winner == "2P")
    n_unknown = sum(1 for w in results if w.winner is None)
    print(f"\n結果: 1P={n_1p}  2P={n_2p}  UNKNOWN={n_unknown}")
    print(f"出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
