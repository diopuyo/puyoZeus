"""STEP A: drain 発火実挙動の短区間診断スクリプト。

t=0〜70s の video_124_4min.mp4 で処理。
state 遷移があるフレームのみ表示し、旧 drain vs tsumo_count増分 を比較する。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console  # noqa: E402
init_console()

import cv2  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO_PATH = Path("data/frames/video_124_4min.mp4")
END_SEC = 70.0
DEFAULT_FPS = 30.0
# サンプリング: 毎フレームは重いので 3 フレームに 1 回 (= 10fps 相当)
SAMPLE_EVERY = 3


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[ERROR] cannot open {VIDEO_PATH}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(total_frames, int(END_SEC * fps))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[input] {VIDEO_PATH} {src_w}x{src_h} fps={fps:.1f} frames={n_frames}")
    print(f"[range] t=0-{END_SEC}s, sample_every={SAMPLE_EVERY}")

    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
    )
    pipeline.set_video_id("video_124_4min")

    ojama_tracker = OjamaAccountingTracker()
    ojama_tracker.reset()

    prev_state_p1 = BoardState.MENU
    prev_state_p2 = BoardState.MENU
    prev_tsumo_p1 = 0
    prev_tsumo_p2 = 0

    # カウンタ
    tsumo_fall_to_stable_p1 = 0
    tsumo_fall_to_stable_p2 = 0
    tsumo_count_drain_p1 = 0
    tsumo_count_drain_p2 = 0

    snap = ojama_tracker.get_snapshot(0.0)
    prev_forecast_p2 = -1
    prev_forecast_p1 = -1

    print(f"\n{'t_sec':>8} {'1P_state':>14} {'2P_state':>14} "
          f"{'tc1P':>5} {'tc2P':>5} "
          f"{'Δ1P':>4} {'Δ2P':>4} "
          f"{'dr1P':>5} {'dr2P':>5} "
          f"{'fc1P':>5} {'fc2P':>5}")
    print("-" * 100)

    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t_sec = fi / fps

        # サンプリング間引き
        if fi % SAMPLE_EVERY != 0:
            continue

        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)

        result = pipeline.update(fi, t_sec, frame)
        p1 = result.p1
        p2 = result.p2

        # 旧 drain 判定: TSUMO_FALL → STABLE 遷移
        old_drain_p1 = (prev_state_p1 == BoardState.TSUMO_FALL and p1.state == BoardState.STABLE)
        old_drain_p2 = (prev_state_p2 == BoardState.TSUMO_FALL and p2.state == BoardState.STABLE)

        ojama_tracker.on_state_transition("p1", prev_state_p1, p1.state, p1.score, t_sec)
        ojama_tracker.on_state_transition("p2", prev_state_p2, p2.state, p2.score, t_sec)

        # 旧挙動: TSUMO_FALL→STABLE でのみ drain
        if old_drain_p1:
            ojama_tracker.on_tsumo_settled("p1", t_sec)
            tsumo_fall_to_stable_p1 += 1
        if old_drain_p2:
            ojama_tracker.on_tsumo_settled("p2", t_sec)
            tsumo_fall_to_stable_p2 += 1

        snap = ojama_tracker.get_snapshot(t_sec)

        # tsumo_count 増分
        cur_tc_p1 = pipeline.tsumo_count("1P")
        cur_tc_p2 = pipeline.tsumo_count("2P")
        delta_p1 = max(0, cur_tc_p1 - prev_tsumo_p1)
        delta_p2 = max(0, cur_tc_p2 - prev_tsumo_p2)
        if delta_p1 > 0:
            tsumo_count_drain_p1 += delta_p1
        if delta_p2 > 0:
            tsumo_count_drain_p2 += delta_p2

        fc_p1 = snap.forecast_p1
        fc_p2 = snap.forecast_p2

        # 変化行のみ表示
        if (
            fc_p1 != prev_forecast_p1 or fc_p2 != prev_forecast_p2
            or old_drain_p1 or old_drain_p2
            or delta_p1 > 0 or delta_p2 > 0
            or p1.state != prev_state_p1 or p2.state != prev_state_p2
        ):
            print(
                f"{t_sec:8.2f} {p1.state.value:>14} {p2.state.value:>14} "
                f"{cur_tc_p1:5d} {cur_tc_p2:5d} "
                f"{delta_p1:+4d} {delta_p2:+4d} "
                f"{'Y' if old_drain_p1 else '-':>5} {'Y' if old_drain_p2 else '-':>5} "
                f"{fc_p1:5d} {fc_p2:5d}"
            )

        prev_forecast_p1 = fc_p1
        prev_forecast_p2 = fc_p2
        prev_state_p1 = p1.state
        prev_state_p2 = p2.state
        prev_tsumo_p1 = cur_tc_p1
        prev_tsumo_p2 = cur_tc_p2

    cap.release()

    print("\n" + "=" * 60)
    print("=== 集計 ===")
    print(f"  旧drain(TSUMO_FALL→STABLE) 1P={tsumo_fall_to_stable_p1}  2P={tsumo_fall_to_stable_p2}")
    print(f"  tsumo_count増分累積         1P={tsumo_count_drain_p1}  2P={tsumo_count_drain_p2}")
    if tsumo_count_drain_p1 > 0:
        ratio = tsumo_fall_to_stable_p1 / tsumo_count_drain_p1
        print(f"  1P倍率: {tsumo_fall_to_stable_p1}/{tsumo_count_drain_p1} = {ratio:.1f}x")
    if tsumo_count_drain_p2 > 0:
        ratio = tsumo_fall_to_stable_p2 / tsumo_count_drain_p2
        print(f"  2P倍率: {tsumo_fall_to_stable_p2}/{tsumo_count_drain_p2} = {ratio:.1f}x")
    print(f"最終 forecast: 1P={snap.forecast_p1} 2P={snap.forecast_p2}")  # type: ignore[possibly-undefined]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
