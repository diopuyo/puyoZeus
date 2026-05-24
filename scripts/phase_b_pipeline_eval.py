"""Phase B-8: BoardStateMachine pipeline を 1 動画で回し統計を出力。

出力サマリ:
    - 全 frame 数, frame レート
    - 各 state の frame 数 (1P/2P 別、絶対 + 比率)
    - STABLE 確定盤面ユニーク数 (= 確定の安定性)
    - drift 検出 frame 数 / re-sync 回数
    - 平均 mismatch_count (drift detector 出力)
    - 連鎖検出回数

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_b_pipeline_eval \
        --video data/frames/video_06.mp4 \
        --start-sec 385 --end-sec 415

このスクリプトは GT 不要。pipeline が動画を最後まで回せるか + state 分布が
妥当な範囲に収まるかを smoke test 的に確認するもの。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402

init_console()

import cv2  # noqa: E402

from src.board_state_machine import BoardState  # noqa: E402
from src.recognition_pipeline import (  # noqa: E402
    PipelineResult, RecognitionPipeline, SideResult,
)


# ============================
# 統計
# ============================


@dataclass
class SideStats:
    side: str
    state_counts: Counter[BoardState] = field(default_factory=Counter)
    drift_frames: int = 0
    resync_frames: int = 0
    total_mismatch: int = 0
    chain_event_count: int = 0
    unique_confirmed: set[bytes] = field(default_factory=set)

    def update(self, side_result: SideResult) -> None:
        self.state_counts[side_result.state] += 1
        if side_result.drift.is_drift:
            self.drift_frames += 1
        if side_result.drift.needs_resync:
            self.resync_frames += 1
        self.total_mismatch += side_result.drift.mismatch_count
        if side_result.chain_event is not None:
            self.chain_event_count += 1
        if side_result.confirmed_board is not None:
            # bytes 化してユニーク化
            self.unique_confirmed.add(
                side_result.confirmed_board.to_json().encode("utf-8")
            )


def summarize(stats: SideStats, total_frames: int) -> str:
    lines = [f"=== {stats.side} ==="]
    for st in BoardState:
        n = stats.state_counts.get(st, 0)
        ratio = 100.0 * n / total_frames if total_frames else 0.0
        lines.append(f"  {st.value:<12} {n:6d}  ({ratio:5.1f}%)")
    lines.append(
        f"  drift_frames    {stats.drift_frames:6d}"
    )
    lines.append(
        f"  resync_frames   {stats.resync_frames:6d}"
    )
    avg_mm = stats.total_mismatch / total_frames if total_frames else 0.0
    lines.append(
        f"  avg_mismatch    {avg_mm:6.2f}"
    )
    lines.append(
        f"  chain_events    {stats.chain_event_count:6d}"
    )
    lines.append(
        f"  unique_confirmed {len(stats.unique_confirmed):5d}"
    )
    return "\n".join(lines)


# ============================
# main
# ============================


def evaluate_video(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    fps_sample: float,
    stable_frame_count: int,
    no_chain: bool,
    no_score: bool,
) -> int:
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        return 1

    print(f"[eval] video = {to_windows_path(video_path)}")
    print(f"[eval] range = [{start_sec:.2f}, {end_sec:.2f}] sec")
    print(f"[eval] fps_sample = {fps_sample:.1f}")
    print(
        f"[eval] stable_n = {stable_frame_count}, "
        f"chain_tracker = {not no_chain}, score_ocr = {not no_score}"
    )

    pipe = RecognitionPipeline.load_default(
        stable_frame_count=stable_frame_count,
        load_score_ocr=not no_score,
        enable_chain_tracker=not no_chain,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: video open failed: {video_path}")
        return 1

    interval = 1.0 / fps_sample
    p1_stats = SideStats(side="1P")
    p2_stats = SideStats(side="2P")
    total = 0
    t = start_sec
    frame_idx = 0
    while t < end_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[eval] read failed at t={t:.2f}, stopping")
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        result: PipelineResult = pipe.update(frame_idx, t, frame)
        p1_stats.update(result.p1)
        p2_stats.update(result.p2)
        total += 1
        if frame_idx % 50 == 0:
            print(
                f"[eval] frame {frame_idx:4d}  t={t:7.2f}  "
                f"1P={result.p1.state.value} 2P={result.p2.state.value}  "
                f"drift_1P={result.p1.drift.mismatch_count} "
                f"drift_2P={result.p2.drift.mismatch_count}"
            )
        frame_idx += 1
        t += interval
    cap.release()

    print()
    print(f"[eval] total frames = {total}")
    print(summarize(p1_stats, total))
    print(summarize(p2_stats, total))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument(
        "--fps-sample", type=float, default=20.0,
        help="サンプリング fps (state machine 入力 frame レート)",
    )
    parser.add_argument(
        "--stable-frame-count", type=int, default=6,
        help="STABLE 確定の連続多数決 frame 数",
    )
    parser.add_argument("--no-chain", action="store_true")
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args()

    return evaluate_video(
        video_path=args.video,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        fps_sample=args.fps_sample,
        stable_frame_count=args.stable_frame_count,
        no_chain=args.no_chain,
        no_score=args.no_score,
    )


if __name__ == "__main__":
    sys.exit(main())
