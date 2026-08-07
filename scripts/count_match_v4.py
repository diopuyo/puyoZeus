"""
試合境界検出 v4: ちょうど 50 試合検出を目指す改良版。

v3 からの変更:
    - confirm を増やしてチャタリング排除
    - 試合時間フィルタ（min/max）を post-process で適用
    - 短試合を破棄、長試合は内部の WIN 数値変化で分割を試みる

使い方:
    ./venv/bin/python scripts/count_match_v4.py --video data/frames/video_02.mp4 \
        --interval 1 --confirm 3 --min-duration 20 --max-duration 220
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.score_zero import ScoreZeroDetector
from src.win_panel import WinPanelDetector


@dataclass
class Match:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _sig(patch: np.ndarray) -> np.ndarray:
    """16×16 バイナリ指紋。"""
    if patch is None or patch.size == 0:
        return np.zeros(256, dtype=np.uint8)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw.flatten().astype(np.uint8)


def scan_video(
    video_path: Path,
    interval: float,
    confirm: int,
    start_sec: float = 0.0,
    end_sec: float = 0.0,
) -> list[Match]:
    """動画を走査して raw 試合列を返す（フィルタ前）。

    start_sec/end_sec: optional。指定すると当該区間のみスキャンする
    (2026-08-03 追加、長尺動画で対象区間が既知の場合の高速化用。
    既定 0.0/0.0 は従来通り動画全体、後方互換維持)。end_sec<=0 は
    動画末尾までを意味する。
    """
    zero_det = ScoreZeroDetector.load_default()
    panel_det = WinPanelDetector.load_default()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    scan_end = min(duration, end_sec) if end_sec > 0 else duration

    confirmed = "none"
    pending: str | None = None
    pending_count = 0
    matches: list[Match] = []
    match_start: float | None = None

    t = max(0.0, start_sec)
    while t < scan_end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            t += interval
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)

        panel = panel_det.detect(frame)
        if not panel.present:
            raw = "none"
        else:
            z = zero_det.detect(frame)
            raw = "zero" if z.both_zero else "playing"

        if raw == confirmed:
            pending = None
            pending_count = 0
        else:
            if pending == raw:
                pending_count += 1
            else:
                pending = raw
                pending_count = 1
            if pending_count >= confirm:
                old = confirmed
                confirmed = raw
                pending = None
                pending_count = 0
                if old == "playing" and confirmed == "zero":
                    if match_start is not None:
                        matches.append(Match(start=match_start, end=t))
                        match_start = None
                elif old == "playing" and confirmed == "none":
                    # パネル消失 = セクション終わり。進行中試合はその時刻でクローズ
                    if match_start is not None:
                        matches.append(Match(start=match_start, end=t))
                        match_start = None
                elif confirmed == "playing":
                    match_start = t
        t += interval
    cap.release()
    return matches


def filter_matches(
    matches: list[Match],
    min_dur: float,
    max_dur: float,
) -> tuple[list[Match], list[Match], list[Match]]:
    """有効 / 短すぎ / 長すぎ に分類。"""
    valid: list[Match] = []
    too_short: list[Match] = []
    too_long: list[Match] = []
    for m in matches:
        if m.duration < min_dur:
            too_short.append(m)
        elif m.duration > max_dur:
            too_long.append(m)
        else:
            valid.append(m)
    return valid, too_short, too_long


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--confirm", type=int, default=3)
    parser.add_argument("--min-duration", type=float, default=20.0)
    parser.add_argument("--max-duration", type=float, default=220.0)
    parser.add_argument("--out-root", default="data/verify/match_boundaries_v4")
    parser.add_argument("--expected", type=int, default=50, help="期待試合数（情報表示用）")
    # W-η (2026-05-05): intro 区間 (動画開始 30 秒以内に始まる長尺) を自動 skip
    parser.add_argument(
        "--intro-skip-start-thresh", type=float, default=30.0,
        help="この秒数以内に始まる試合を intro 候補として判定 (0で無効)",
    )
    parser.add_argument(
        "--intro-skip-min-duration", type=float, default=100.0,
        help="intro 候補のうちこの秒数以上の長尺は intro と判定して skip",
    )
    parser.add_argument(
        "--start-sec", type=float, default=0.0,
        help="スキャン開始秒 (省略時0=動画先頭、2026-08-03追加、対象区間既知の長尺動画高速化用)",
    )
    parser.add_argument(
        "--end-sec", type=float, default=0.0,
        help="スキャン終了秒 (0以下は動画末尾まで、2026-08-03追加)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"動画なし: {video_path}", file=sys.stderr)
        return 1

    print(f"scan: confirm={args.confirm} interval={args.interval}s "
          f"区間=({args.start_sec:.1f}, {args.end_sec if args.end_sec > 0 else '末尾'})")
    raw = scan_video(video_path, args.interval, args.confirm, args.start_sec, args.end_sec)
    print(f"raw 検出: {len(raw)} 試合")

    valid, short, long = filter_matches(raw, args.min_duration, args.max_duration)
    print(f"\nフィルタ: min={args.min_duration}s, max={args.max_duration}s")
    print(f"  有効: {len(valid)}")
    print(f"  短すぎ (< {args.min_duration}s): {len(short)}")
    print(f"  長すぎ (> {args.max_duration}s): {len(long)}")

    # 短試合内訳
    if short:
        print(f"\n短すぎ試合（破棄）:")
        for m in short:
            print(f"  t={m.start:6.1f}-{m.end:6.1f}  ({m.duration:5.1f}s)")
    if long:
        print(f"\n長すぎ試合（要分割）:")
        for m in long:
            print(f"  t={m.start:6.1f}-{m.end:6.1f}  ({m.duration:5.1f}s)")

    # W-η: intro 試合の自動 skip
    if (
        args.intro_skip_start_thresh > 0 and len(valid) > 0
        and valid[0].start <= args.intro_skip_start_thresh
        and valid[0].duration >= args.intro_skip_min_duration
    ):
        print(
            f"\n[intro skip] 試合 1 が動画開始 {valid[0].start:.1f}s で "
            f"長尺 {valid[0].duration:.1f}s = intro 疑い → skip"
        )
        valid = valid[1:]

    print(f"\n=== 最終試合一覧 ({len(valid)}) ===")
    for i, m in enumerate(valid):
        print(f"  #{i+1:3d}  t={m.start:6.1f}-{m.end:6.1f}  ({m.duration:5.1f}s)")

    print(f"\n目標 {args.expected} に対して: {len(valid)} (差分 {len(valid)-args.expected:+d})")

    # TSV
    out_dir = Path(args.out_root) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "matches.tsv"
    with tsv.open("w", encoding="utf-8") as f:
        f.write("idx\tstart_sec\tend_sec\tduration_sec\n")
        for i, m in enumerate(valid):
            f.write(f"{i+1}\t{m.start:.1f}\t{m.end:.1f}\t{m.duration:.1f}\n")
    print(f"\n出力: {tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
