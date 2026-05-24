"""W4: score_ocr の実値で試合境界を検出する代替実装。

count_match_v4 (score_zero detector ベース) が pl5/pl6 動画で機能しないため、
ScoreOcr で読み取った実値 (両側 score=0 信頼度高) で試合間を判定する。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.count_match_via_ocr \
        --video data/frames/video_10.mp4 \
        --interval 1.0 --confirm 2 --min-duration 20 --max-duration 220 \
        --out data/verify/match_boundaries_v5/video_10/matches.tsv
"""
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

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2

from src.score_ocr import ScoreOcr


def is_zero_state(
    score_1p: int | None, conf_1p: float,
    score_2p: int | None, conf_2p: float,
    conf_threshold: float = 0.5,
) -> bool:
    """両側 score=0 + 高信頼度なら True。"""
    if conf_1p < conf_threshold or conf_2p < conf_threshold:
        return False
    if score_1p is None or score_2p is None:
        return False
    return score_1p == 0 and score_2p == 0


def scan_video(
    video_path: Path,
    interval: float,
    confirm: int,
    conf_threshold: float = 0.5,
) -> list[tuple[float, float]]:
    """動画を interval 秒間隔でスキャン、両側 zero の連続範囲を試合間と認定。

    試合間 (zero=True 連続) と試合中 (zero=False 連続) の境界を取り、
    試合中の区間 (start_sec, end_sec) リストを返す。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, cap.get(cv2.CAP_PROP_FPS)))

    ocr = ScoreOcr.load_default()
    samples: list[tuple[float, bool]] = []  # (t, is_zero)

    t = 0.0
    while t <= total_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if ok and fr is not None:
            if fr.shape[:2] != (1080, 1920):
                fr = cv2.resize(fr, (1920, 1080), interpolation=cv2.INTER_AREA)
            r = ocr.read(fr)
            zero = is_zero_state(
                r.score_1p, r.confidence_1p,
                r.score_2p, r.confidence_2p,
                conf_threshold,
            )
            samples.append((t, zero))
        t += interval

    cap.release()

    # confirm 連続 で zero/in-match を確定
    matches: list[tuple[float, float]] = []
    in_match = False
    in_match_start: float = 0.0
    zero_streak = 0
    nonzero_streak = 0

    for t_sec, is_zero in samples:
        if is_zero:
            zero_streak += 1
            nonzero_streak = 0
            if in_match and zero_streak >= confirm:
                # 試合終了確定
                end_sec = t_sec - (confirm - 1) * interval
                matches.append((in_match_start, end_sec))
                in_match = False
        else:
            nonzero_streak += 1
            zero_streak = 0
            if not in_match and nonzero_streak >= confirm:
                in_match_start = t_sec - (confirm - 1) * interval
                in_match = True

    # 動画末尾で試合中なら最後まで含める
    if in_match and samples:
        matches.append((in_match_start, samples[-1][0]))

    return matches


def filter_matches(
    matches: list[tuple[float, float]],
    min_duration: float,
    max_duration: float,
) -> tuple[list, list, list]:
    valid = []
    short = []
    long_ = []
    for s, e in matches:
        d = e - s
        if d < min_duration:
            short.append((s, e, d))
        elif d > max_duration:
            long_.append((s, e, d))
        else:
            valid.append((s, e, d))
    return valid, short, long_


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--confirm", type=int, default=2)
    parser.add_argument("--min-duration", type=float, default=20.0)
    parser.add_argument("--max-duration", type=float, default=220.0)
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    video_path = Path(args.video)
    print(f"scan: {video_path.name} interval={args.interval} confirm={args.confirm}")
    matches = scan_video(
        video_path, args.interval, args.confirm,
        conf_threshold=args.conf_threshold,
    )
    print(f"raw matches: {len(matches)}")

    valid, short, long_ = filter_matches(
        matches, args.min_duration, args.max_duration,
    )
    print(f"  valid: {len(valid)}, short: {len(short)}, long: {len(long_)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["idx", "start_sec", "end_sec", "duration_sec"])
        for i, (s, e, d) in enumerate(valid, 1):
            w.writerow([i, f"{s:.1f}", f"{e:.1f}", f"{d:.1f}"])
    print(f"saved: {to_windows_path(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
