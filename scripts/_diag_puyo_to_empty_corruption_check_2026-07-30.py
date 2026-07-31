"""色->空 HSVガード(enable_puyo_to_empty_hsv_guard)の corruption 計測 (read-only診断, 2026-07-30)。

measure_stable_cell_acc.py の「後処理破壊検知」(postprocess_corruption) と同一定義を
流用した簡易版: raw_cnn(CNN+HSVハイブリッドの直接出力)==raw_hsv(HSV-onlyの直接出力)
なのに confirmed(状態機械の確定盤面)が異なるセルを corruption として数える。
特に色->空 (raw==色, confirmed==EMPTY) のサブカテゴリが本ガードの直接標的。

注意 (重要な限界):
  measure_stable_cell_acc.py が使う「正解ラベル」は外部の人手ラベルではなく、
  raw_cnn/raw_hsv/confirmed の 3 者多数決による自己無矛盾性チェックである
  (2/3 以上一致で正解確定)。本スクリプトはそのうち corruption 検知部分のみを
  切り出した簡易版で、真の正解率(ground truth accuracy)を測るものではない。
  raw_cnn と raw_hsv が「同じ誤り」に合意した場合は検知できない(fail-silent)。

制約: read-only診断。src/は一切変更しない。cv2.setNumThreads(1)、並列しない。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402

VIDEO_DIR = PROJ_ROOT / "data" / "frames"


def _print(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(video: str, start_sec: float, max_sec: float, guard_on: bool) -> dict:
    cv2.setNumThreads(1)
    path = VIDEO_DIR / f"video_{video}.mp4"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"開けません: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + max_sec) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    pipe_cnn = RecognitionPipeline.load_default(
        enable_puyo_to_empty_hsv_guard=guard_on,
    )
    pipe_cnn.set_video_id(video)
    pipe_hsv = RecognitionPipeline.load_default(cnn_override_prob=2.0)
    pipe_hsv.set_video_id(video)

    n_stable = 0
    n_total_cells = 0
    n_corruption = 0
    n_color_to_empty = 0
    n_color_to_color = 0
    n_empty_to_color = 0

    fi = start_frame
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t = fi / fps
        r_cnn = pipe_cnn.update(fi, t, frame)
        r_hsv = pipe_hsv.update(fi, t, frame)
        for sr_cnn, sr_hsv in ((r_cnn.p1, r_hsv.p1), (r_cnn.p2, r_hsv.p2)):
            if sr_cnn.state != BoardState.STABLE or sr_cnn.confirmed_board is None:
                continue
            n_stable += 1
            for row in range(BOARD_ROWS):
                for col in range(BOARD_COLS):
                    raw_cnn_val = int(sr_cnn.cnn_board.get(row, col))
                    raw_hsv_val = int(sr_hsv.cnn_board.get(row, col))
                    confirmed_val = int(sr_cnn.confirmed_board.get(row, col))
                    if COLOR_UNKNOWN in (raw_cnn_val, raw_hsv_val, confirmed_val):
                        continue
                    n_total_cells += 1
                    if raw_cnn_val == raw_hsv_val and confirmed_val != raw_cnn_val:
                        n_corruption += 1
                        if raw_cnn_val == COLOR_EMPTY and confirmed_val != COLOR_EMPTY:
                            n_empty_to_color += 1
                        elif raw_cnn_val != COLOR_EMPTY and confirmed_val == COLOR_EMPTY:
                            n_color_to_empty += 1
                        else:
                            n_color_to_color += 1
        fi += 1

    cap.release()
    return {
        "video": video, "guard_on": guard_on, "fps": fps,
        "n_stable_side_frames": n_stable, "n_total_cells": n_total_cells,
        "n_corruption": n_corruption,
        "n_color_to_empty": n_color_to_empty,
        "n_color_to_color": n_color_to_color,
        "n_empty_to_color": n_empty_to_color,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-sec", type=float, required=True)
    ap.add_argument("--max-sec", type=float, required=True)
    ap.add_argument("--guard", choices=["off", "on"], required=True)
    args = ap.parse_args()
    _print(f"開始 video={args.video} guard={args.guard}")
    result = run(args.video, args.start_sec, args.max_sec, args.guard == "on")
    _print(f"完了: {result}")
    print(result)


if __name__ == "__main__":
    main()
