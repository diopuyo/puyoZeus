"""持続誤認26件系統1 修正の実データ不発火の原因診断 (2026-08-17)。

c23 (t=1405.12, r7c1等) の実映像を再生し、以下を毎フレーム計装する:
    - cycle 71n の override が実際に発火したか (どのセルをいつ)
    - override 発火時に watch リストへ登録されたか
    - watch リスト再チェックで CNN==HSV 一致が得られたか (得られない場合その理由)
    - CNN 生観測がいつ正解に戻ったか (h_list の中身)

本体コード変更なし (モンキーパッチで計装するのみ)。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(_ROOT))

from src.recognition_pipeline import RecognitionPipeline
from src.board_state_machine import BoardState

# 計装: _validate_next_history の呼び出しをフックし、対象セルの
# next_queue/ever_seen 状態と入出力を出力する (本体コード変更なし)。
_orig_validate = RecognitionPipeline._validate_next_history


def _patched_validate(board, next_queue, ever_seen=None, frame_bgr=None, region=None):
    out = _orig_validate(board, next_queue, ever_seen=ever_seen, frame_bgr=frame_bgr, region=region)
    r, c = TARGET_CELL
    before_v = int(board.get(r, c))
    after_v = int(out.get(r, c))
    if before_v != after_v:
        print(
            f"  [_validate_next_history] cell({r},{c}) {before_v}->{after_v} "
            f"next_queue={next_queue} ever_seen={ever_seen}"
        )
    return out


RecognitionPipeline._validate_next_history = staticmethod(_patched_validate)

VIDEO = Path.home() / "frames" / "video_c10.mp4"
TARGET_CELL = (4, 1)
TARGET_SIDE = "2P"
# 元診断: t=1405.12 で一斉転落。前後 ±3秒を計装。
START_SEC = 1337.4
END_SEC = 1412.0


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"fps={fps}")
    start_frame = int(START_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        load_next_detector=True,
        temporal_smoothing=1,
        force_in_match=True,
        enable_effect_gate=True,
        enable_burst_guard_v2=True,
        enable_transition_merge_guard=True,
        burst_gate_open_threshold=0.954,
        enable_hidden_row_burst_guard=True,
        enable_match_transition_debounce=True,
        enable_ojama_fall_placement_override=True,
        enable_patch_fp_hsv_guard=True,
        enable_chain_tracker=True,
        enable_floating_gap_restore=True,
        enable_landing_color_guard=True,
        enable_override_color_guard=True,
        enable_ojama_column_stack_fix=True,
    )
    r, c = TARGET_CELL
    prev_hist_len = -1
    prev_watch = None
    prev_val = None
    frame_idx = start_frame
    t_sec = START_SEC
    while t_sec < END_SEC:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        result = pipeline.update(frame_idx, t_sec, frame)
        side_result = result.p2 if TARGET_SIDE == "2P" else result.p1
        cb = side_result.confirmed_board
        cur_val = int(cb.get(r, c)) if cb is not None else None
        hist = (
            pipeline._stable_cnn_history_2p if TARGET_SIDE == "2P"
            else pipeline._stable_cnn_history_1p
        )
        watch = (
            pipeline._landing_color_watch_2p if TARGET_SIDE == "2P"
            else pipeline._landing_color_watch_1p
        )
        h_list = hist.get((r, c), [])
        cnn_v = int(side_result.cnn_board.get(r, c)) if side_result.cnn_board is not None else None
        changed = (cur_val != prev_val)
        watch_changed = (watch != prev_watch)
        if changed or watch_changed or len(h_list) != prev_hist_len:
            print(
                f"t={t_sec:.3f} frame={frame_idx} state={side_result.state} "
                f"confirmed={cur_val} cnn_raw={cnn_v} hist_len={len(h_list)} "
                f"hist_tail={h_list[-5:]} watch={watch}"
            )
        prev_val = cur_val
        prev_hist_len = len(h_list)
        prev_watch = list(watch)
        frame_idx += 1
        t_sec = frame_idx / fps
    cap.release()


if __name__ == "__main__":
    main()
