"""
スコア「00000000」検出を利用した試合境界カウント。

ロジック:
    - 1 秒間隔でフレーム走査
    - ScoreZeroDetector + WinPanelDetector で状態確定
    - 状態 = "both_zero"（両側 00000000）: 試合の合間
    - 状態 = "playing"（少なくとも片方非ゼロ）: 試合中
    - "playing" → "both_zero" 遷移を試合終了とカウント
    - 連続一致（CONFIRM 回）で確定してチャタリング排除

出力:
    stdout: 試合数 + 各試合の秒数範囲
    data/verify/match_boundaries_v3/<video>/match_end_XXXXXs_{before,after}.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.score_zero import ScoreZeroDetector
from src.win_panel import WinPanelDetector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--confirm", type=int, default=2,
                        help="状態確定に必要な連続一致数")
    parser.add_argument("--out-root", default="data/verify/match_boundaries_v3")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"動画なし: {video_path}", file=sys.stderr)
        return 1

    zero_det = ScoreZeroDetector.load_default()
    panel_det = WinPanelDetector.load_default()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    print(f"動画: {video_path.name}  duration={duration:.0f}s")

    out_dir = Path(args.out_root) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 状態: "none" (パネルなし) / "zero" (両側00000000) / "playing" (少なくとも片方非ゼロ)
    confirmed = "none"
    pending: str | None = None
    pending_count = 0
    prev_frame: np.ndarray | None = None
    transitions: list[tuple[str, float, np.ndarray | None, np.ndarray]] = []
    match_ranges: list[tuple[float, float]] = []   # (start, end) 各試合
    match_start_t: float | None = None

    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            t += args.interval
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
            if pending_count >= args.confirm:
                old = confirmed
                confirmed = raw
                pending = None
                pending_count = 0
                ev = f"{old}->{confirmed}"
                transitions.append((ev, t, prev_frame, frame.copy()))
                # 試合カウント
                if old == "playing" and confirmed == "zero":
                    # 試合終了
                    if match_start_t is not None:
                        match_ranges.append((match_start_t, t))
                        match_start_t = None
                elif old == "zero" and confirmed == "playing":
                    match_start_t = t
                elif old == "none" and confirmed == "playing":
                    match_start_t = t

        prev_frame = frame.copy()
        t += args.interval

    cap.release()

    # 保存
    for ev, ts, before, after in transitions:
        if "->" not in ev:
            continue
        before_state, after_state = ev.split("->")
        if before_state == "playing" and after_state == "zero":
            kind = "match_end"
        elif before_state == "zero" and after_state == "playing":
            kind = "match_start"
        elif after_state == "none":
            kind = "section_end"
        elif before_state == "none":
            kind = "section_start"
        else:
            kind = ev.replace("->", "_to_")
        tag = f"{kind}_{int(ts):05d}s"
        if before is not None:
            cv2.imwrite(str(out_dir / f"{tag}_before.png"), before)
        cv2.imwrite(str(out_dir / f"{tag}_after.png"), after)

    n_end = sum(1 for e in transitions if e[0] == "playing->zero")
    n_start = sum(1 for e in transitions if e[0] == "zero->playing")
    print(f"\n試合終了検出: {n_end}")
    print(f"試合開始検出: {n_start}")
    print(f"確定試合範囲: {len(match_ranges)}")
    for i, (s, e) in enumerate(match_ranges):
        print(f"  #{i+1:3d}  t={s:6.1f}-{e:6.1f}  ({e-s:5.1f}s)")
    print(f"\n出力: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
