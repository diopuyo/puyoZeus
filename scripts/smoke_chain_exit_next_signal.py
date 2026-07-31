"""案X*(A)(B)+warmup スモークテスト: v89_match01 で chain span を計測する。

OFF/ON それぞれで pipeline を動かし、1P の連鎖保持 span を比較する。
v89 1P: t≈33.77 発火 → t≈40.87 まで 6.87 秒 CHAIN 保持 が期待値 (OFF)。
ON 時: 次ツモ slide 検知で連鎖が早期終了し、span が短縮されることを確認する。

Usage:
    PYTHONPATH=. venv/bin/python scripts/smoke_chain_exit_next_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
_proj = Path(__file__).resolve().parent.parent
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

import cv2
import numpy as np
from src.recognition_pipeline import RecognitionPipeline
from src.board_state_machine import BoardState

# 評価対象動画
_VIDEO_PATH = _proj / "data/match_clips/v89/v89_match01.mp4"
# 期待する v89 1P 連鎖発火時刻の付近 (秒)
_CHAIN_START_T_EXPECTED = 33.0
_CHAIN_END_T_EXPECTED_OFF = 40.87  # OFF 時の過剰保持終了時刻
# span が大幅に短縮 (≤ 5.0s 以内) されたら改善と判定
_IMPROVED_SPAN_THRESHOLD = 5.0


def _measure_chain_spans(
    video_path: Path,
    enable_next_signal: bool,
    max_sec: float = 50.0,
) -> list[tuple[float, float, str]]:
    """pipeline を動かして CHAIN state の span リストを返す。

    Returns:
        list of (chain_start_sec, chain_end_sec, side)
    """
    pipe = RecognitionPipeline.load_default(
        force_in_match=True,
        enable_chain_exit_next_signal=enable_next_signal,
        enable_chain_formula_detection=True,
    )
    _vid_id = "v89"
    pipe.set_video_id(_vid_id)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[smoke] 動画を開けません: {video_path}", file=sys.stderr)
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    spans: list[tuple[float, float, str]] = []
    chain_start_1p: float | None = None
    chain_start_2p: float | None = None
    prev_state_1p: BoardState = BoardState.STABLE
    prev_state_2p: BoardState = BoardState.STABLE

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        if t > max_sec:
            break

        result = pipe.update(frame_idx, t, frame)
        if result is None:
            frame_idx += 1
            continue

        # 1P
        cur_1p = result.p1.state
        if prev_state_1p != BoardState.CHAIN and cur_1p == BoardState.CHAIN:
            chain_start_1p = t
        elif prev_state_1p == BoardState.CHAIN and cur_1p != BoardState.CHAIN:
            if chain_start_1p is not None:
                spans.append((chain_start_1p, t, "1P"))
                chain_start_1p = None
        prev_state_1p = cur_1p

        # 2P
        cur_2p = result.p2.state
        if prev_state_2p != BoardState.CHAIN and cur_2p == BoardState.CHAIN:
            chain_start_2p = t
        elif prev_state_2p == BoardState.CHAIN and cur_2p != BoardState.CHAIN:
            if chain_start_2p is not None:
                spans.append((chain_start_2p, t, "2P"))
                chain_start_2p = None
        prev_state_2p = cur_2p

        frame_idx += 1

    # 動画終端で CHAIN のままなら打ち切り
    t_end = frame_idx / fps
    if chain_start_1p is not None:
        spans.append((chain_start_1p, t_end, "1P*"))
    if chain_start_2p is not None:
        spans.append((chain_start_2p, t_end, "2P*"))

    cap.release()
    return spans


def main() -> int:
    if not _VIDEO_PATH.exists():
        print(f"[smoke] 動画が見つかりません: {_VIDEO_PATH}")
        print("[smoke] スモークテストをスキップします")
        return 0

    print(f"[smoke] 対象動画: {_VIDEO_PATH}")
    print()

    # --- OFF ---
    print("[smoke] === OFF (baseline) ===")
    spans_off = _measure_chain_spans(_VIDEO_PATH, enable_next_signal=False)
    print(f"[smoke] CHAIN spans (OFF): {len(spans_off)} 件")
    max_span_off_1p: float = 0.0
    for s, e, side in sorted(spans_off, key=lambda x: x[0]):
        span = e - s
        print(f"  {side}: {s:.2f}s → {e:.2f}s (span={span:.2f}s)")
        if "1P" in side and span > max_span_off_1p:
            max_span_off_1p = span

    print()

    # --- ON ---
    print("[smoke] === ON (案X*) ===")
    spans_on = _measure_chain_spans(_VIDEO_PATH, enable_next_signal=True)
    print(f"[smoke] CHAIN spans (ON): {len(spans_on)} 件")
    max_span_on_1p: float = 0.0
    for s, e, side in sorted(spans_on, key=lambda x: x[0]):
        span = e - s
        print(f"  {side}: {s:.2f}s → {e:.2f}s (span={span:.2f}s)")
        if "1P" in side and span > max_span_on_1p:
            max_span_on_1p = span

    print()

    # --- 判定 ---
    print("[smoke] === 結果サマリ ===")
    print(f"1P 最長 span (OFF): {max_span_off_1p:.2f}s")
    print(f"1P 最長 span (ON):  {max_span_on_1p:.2f}s")
    improved = max_span_on_1p < max_span_off_1p
    short_enough = max_span_on_1p <= _IMPROVED_SPAN_THRESHOLD
    print(f"短縮された: {improved}")
    print(f"閾値以内 ({_IMPROVED_SPAN_THRESHOLD}s): {short_enough}")

    if improved:
        reduction = max_span_off_1p - max_span_on_1p
        print(f"[smoke] PASS: span {max_span_off_1p:.2f}s → {max_span_on_1p:.2f}s (短縮 {reduction:.2f}s)")
        return 0
    else:
        print(f"[smoke] WARN: span が短縮されていない (OFF={max_span_off_1p:.2f}s, ON={max_span_on_1p:.2f}s)")
        print("[smoke] slide signal が検知されなかった可能性。ログ詳細はviz で確認。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
