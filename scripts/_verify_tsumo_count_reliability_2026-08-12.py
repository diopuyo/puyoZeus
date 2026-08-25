"""tsumo_count (RecognitionPipeline.tsumo_count) の信頼性を単体検証する
(2026-08-12、おじゃま収支近似復元 v3 の着地イベントゲート採否判定用)。

data/verify/fps_stride_ab_2026-08-12/ の A/B 実験 (full60fps vs stride2) で
t≈298-304s に tsumo_count が発散 (full60側 9→15 一挙+6) した事象の機構を
特定する。既存 npz は STABLE snapshot ベース (dedup済み) で中間フレームの
状態遷移が見えないため、本スクリプトは動画から直接 RecognitionPipeline を
frame-by-frame で再実行し、以下を計装ログする:
  - BoardState 遷移 (1P/2P 別)
  - pending_tsumo (in-flight キュー) の append/popleft
  - tsumo_count の増分 (フレーム単位)
  - NEXT 変化イベント (pending_tsumo.append の trigger)

stateless 実装原則: 本スクリプトは検証専用の使い捨てツールであり、
src/ 側のロジックは一切変更しない (計装は deque サブクラスでの傍受のみ)。

使い方:
    ./venv/bin/python -m scripts._verify_tsumo_count_reliability_2026-08-12 \\
        --video data/frames/review_demo_2026-08-12.mp4 \\
        --sample-interval-frames 1 \\
        --max-sec 320 \\
        --log-window 285 310 \\
        --out logs/tsumo_verify_full60_2026-08-12.log
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board_state_machine import BoardState  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

TARGET_W: int = 1920
TARGET_H: int = 1080
DEFAULT_FPS: float = 30.0
SIDES = ("1P", "2P")


class _LoggingDeque(deque):
    """pending_tsumo の append/popleft を計装ログするラッパー (検証専用)。

    RecognitionPipeline の実装は一切変更せず、__init__ 後に外部から
    self._pending_tsumo_1p 等をこのクラスのインスタンスに差し替えるだけ
    (stateless 検証原則: 恒久コードへの副作用ゼロ)。
    """

    def __init__(self, label: str, log_fn) -> None:
        super().__init__()
        self._label = label
        self._log_fn = log_fn

    def append(self, item) -> None:  # type: ignore[override]
        super().append(item)
        self._log_fn(f"  [pending+] {self._label} append={item} len={len(self)}")

    def popleft(self):  # type: ignore[override]
        item = super().popleft()
        self._log_fn(f"  [pending-] {self._label} popleft={item} len_after={len(self)}")
        return item


def _make_logger(lines: list[str], enabled_flag: list[bool]):
    """時間窓内のみ実際に書き出すロガーを返す。"""

    def _log(msg: str) -> None:
        if enabled_flag[0]:
            lines.append(msg)

    return _log


def verify(
    video_path: Path,
    sample_interval_frames: int,
    max_sec: float,
    log_window: tuple[float, float],
    out_path: Path,
) -> None:
    """1 動画を frame-by-frame 処理し、tsumo_count 機構を計装ログする。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[verify] cannot open: {video_path}", file=sys.stderr)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(total_frames, int(max_sec * fps)) if max_sec > 0 else total_frames

    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=False,
        temporal_smoothing=1,
        load_next_detector=False,
        force_in_match=True,
    )
    vid_match = __import__("re").search(r"(v\d+|video_\d+|review_demo\S*)", video_path.name)
    if vid_match and hasattr(pipeline, "set_video_id"):
        pipeline.set_video_id(vid_match.group(1))

    lines: list[str] = []
    window_active = [False]
    log = _make_logger(lines, window_active)

    # pending_tsumo を計装ラッパーに差し替え (検証専用、恒久コード不変更)
    pipeline._pending_tsumo_1p = _LoggingDeque("1P", log)
    pipeline._pending_tsumo_2p = _LoggingDeque("2P", log)

    prev_state = {"1P": None, "2P": None}
    prev_tsumo_count = {"1P": 0, "2P": 0}
    prev_next_seen = {"1P": None, "2P": None}

    interval = max(1, sample_interval_frames)
    for frame_idx in range(end_frame):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx % interval != 0:
            continue
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        t_sec = frame_idx / fps
        window_active[0] = log_window[0] <= t_sec <= log_window[1]

        if window_active[0]:
            lines.append(f"--- frame_idx={frame_idx} t={t_sec:.3f}s ---")

        result = pipeline.update(frame_idx, t_sec, frame)
        side_results = {"1P": result.p1, "2P": result.p2}

        for side in SIDES:
            sres = side_results[side]
            state = sres.state
            if window_active[0] and state != prev_state[side]:
                lines.append(
                    f"  [state] {side} {prev_state[side]} -> {state} "
                    f"t={t_sec:.3f}s"
                )
            prev_state[side] = state

            # NEXT 変化イベント (pending_tsumo.append の trigger 元) を可視化
            last_seen_attr = f"_last_seen_next_{'1p' if side == '1P' else '2p'}"
            cur_next = getattr(pipeline, last_seen_attr, None)
            if window_active[0] and cur_next != prev_next_seen[side]:
                lines.append(f"  [next_seen] {side} {prev_next_seen[side]} -> {cur_next}")
            prev_next_seen[side] = cur_next

            tc = pipeline.tsumo_count(side)
            if tc != prev_tsumo_count[side]:
                delta = tc - prev_tsumo_count[side]
                msg = (
                    f"  [tsumo_count] {side} {prev_tsumo_count[side]} -> {tc} "
                    f"(delta={delta:+d}) t={t_sec:.3f}s score={sres.score}"
                )
                if window_active[0]:
                    lines.append(msg)
                elif abs(delta) > 1:
                    # 窓外でも異常ジャンプ (|delta|>1) は常時記録する
                    lines.append(f"[OUT-OF-WINDOW ANOMALY] {msg}")
            prev_tsumo_count[side] = tc

    cap.release()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[verify] wrote {len(lines)} lines -> {out_path}")


def main() -> int:
    """CLI エントリポイント。"""
    parser = argparse.ArgumentParser(description="tsumo_count 信頼性の単体検証")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--sample-interval-frames", type=int, default=1)
    parser.add_argument("--max-sec", type=float, default=0.0)
    parser.add_argument("--log-window", type=float, nargs=2, default=(285.0, 310.0))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(
        args.video, args.sample_interval_frames, args.max_sec,
        tuple(args.log_window), args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
