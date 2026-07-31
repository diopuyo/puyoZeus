"""案P3 スモークテスト: v89_match01 で chain span ON/OFF 比較.

CHAIN state の連続持続区間の最長 span を 1P/2P それぞれ計測し、
フラグ OFF (baseline) と ON (案P3) で比較する。

期待: v89 t34-40.87 の 6.87s 超保持が ON 時に CHAIN_MAX_HOLD_SEC (5.0s) 以内に短縮。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

import cv2

from src.recognition_pipeline import RecognitionPipeline
from src.board_state_machine import BoardState

# 対象動画
VIDEO_PATH = PROJ / "data" / "match_clips" / "v89" / "v89_match01.mp4"
# スモークは冒頭 60 秒のみ (v89m01 は試合 1 で連鎖が多い)
SMOKE_MAX_SEC: float = 60.0


def _measure_chain_spans(
    video_path: Path,
    enable_chain_max_hold_override: bool,
) -> dict:
    """指定フラグで pipeline を構築し CHAIN state の最長持続 span を返す."""
    pipeline = RecognitionPipeline.load_default(
        force_in_match=True,
        enable_game_event_chain_exit=True,
        enable_ojama_visual_detection=True,
        enable_ojama_visual_chain_exit=True,
        enable_chain_formula_detection=True,
        enable_chain_max_hold_override=enable_chain_max_hold_override,
    )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 1P/2P それぞれの chain 開始時刻と最長 span を追跡
    chain_start_1p: float | None = None
    chain_start_2p: float | None = None
    max_span_1p: float = 0.0
    max_span_2p: float = 0.0
    spans_1p: list[float] = []
    spans_2p: list[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        time_sec = frame_idx / fps
        if time_sec > SMOKE_MAX_SEC:
            break

        result = pipeline.update(frame_idx, time_sec, frame)

        # 1P
        s1 = result.p1.state
        if s1 == BoardState.CHAIN:
            if chain_start_1p is None:
                chain_start_1p = time_sec
        else:
            if chain_start_1p is not None:
                span = time_sec - chain_start_1p
                spans_1p.append(span)
                if span > max_span_1p:
                    max_span_1p = span
                chain_start_1p = None

        # 2P
        s2 = result.p2.state
        if s2 == BoardState.CHAIN:
            if chain_start_2p is None:
                chain_start_2p = time_sec
        else:
            if chain_start_2p is not None:
                span = time_sec - chain_start_2p
                spans_2p.append(span)
                if span > max_span_2p:
                    max_span_2p = span
                chain_start_2p = None

        frame_idx += 1

    cap.release()
    return {
        "max_span_1p": max_span_1p,
        "max_span_2p": max_span_2p,
        "spans_1p": spans_1p,
        "spans_2p": spans_2p,
        "frames": frame_idx,
    }


def main() -> None:
    if not VIDEO_PATH.exists():
        print(f"[smoke] 動画未発見: {VIDEO_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"[smoke] 案P3 スモーク: {VIDEO_PATH.name} 先頭 {SMOKE_MAX_SEC}s")
    print(f"[smoke] CHAIN_MAX_HOLD_SEC = {RecognitionPipeline.CHAIN_MAX_HOLD_SEC}s")

    print("\n[smoke] --- OFF (baseline) ---")
    off_res = _measure_chain_spans(VIDEO_PATH, enable_chain_max_hold_override=False)
    print(f"  1P 最長 chain span: {off_res['max_span_1p']:.2f}s")
    print(f"  2P 最長 chain span: {off_res['max_span_2p']:.2f}s")
    print(f"  1P 全 span: {[f'{s:.2f}' for s in off_res['spans_1p']]}")
    print(f"  2P 全 span: {[f'{s:.2f}' for s in off_res['spans_2p']]}")

    print("\n[smoke] --- ON (案P3) ---")
    on_res = _measure_chain_spans(VIDEO_PATH, enable_chain_max_hold_override=True)
    print(f"  1P 最長 chain span: {on_res['max_span_1p']:.2f}s")
    print(f"  2P 最長 chain span: {on_res['max_span_2p']:.2f}s")
    print(f"  1P 全 span: {[f'{s:.2f}' for s in on_res['spans_1p']]}")
    print(f"  2P 全 span: {[f'{s:.2f}' for s in on_res['spans_2p']]}")

    # 判定: ON で MAX_HOLD_SEC 以内に収まっているか
    max_hold = RecognitionPipeline.CHAIN_MAX_HOLD_SEC
    max_on = max(on_res["max_span_1p"], on_res["max_span_2p"])
    max_off = max(off_res["max_span_1p"], off_res["max_span_2p"])

    print("\n[smoke] === 判定 ===")
    print(f"  OFF 最長: {max_off:.2f}s")
    print(f"  ON  最長: {max_on:.2f}s")
    print(f"  MAX_HOLD_SEC: {max_hold:.1f}s")
    if max_on <= max_hold + 0.5:  # 0.5s マージン (frame 境界)
        print(f"  PASS: ON 時の最長 span ({max_on:.2f}s) が MAX_HOLD ({max_hold}s) 以内")
    else:
        print(f"  WARN: ON 時の最長 span ({max_on:.2f}s) が MAX_HOLD ({max_hold}s) を超過")
    if max_on < max_off:
        diff = max_off - max_on
        print(f"  chain 持続が {diff:.2f}s 短縮 (OFF={max_off:.2f}s → ON={max_on:.2f}s)")
    else:
        print("  INFO: chain span の短縮効果なし (v89m01 に 5s 超の連鎖がない可能性)")


if __name__ == "__main__":
    main()
