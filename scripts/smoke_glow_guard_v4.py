"""スモークテスト: 予告おじゃま発光ガード v4 (glow guard OFF vs v4).

v89_match01 の t=66-72s (1P 上部全セル row0-4 × col0-5) で
おじゃま誤認(値9) frame 総数を比較する。

使い方:
  PYTHONPATH=. venv/bin/python scripts/smoke_glow_guard_v4.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from src.board import COLOR_OJAMA, BOARD_ROWS, BOARD_COLS
from src.recognition_pipeline import RecognitionPipeline

# ===== 設定 =====
_CLIP = _PROJ / "data" / "match_clips" / "v89" / "v89_match01.mp4"
# スモーク対象: 発火帯 t=66-72s
_T_START = 66.0
_T_END   = 72.0
# 1P 上部 row 範囲 (row0-4: 0行目 = 隠し段, 1-4 = 上部可視行)
_ROW_RANGE = range(0, 5)
_COL_RANGE = range(0, BOARD_COLS)  # 0-5 全列


def _count_upper_ojama(confirmed: Any, side: str = "1P") -> int:
    """1P 上部 row0-4 のおじゃまセル数を返す。"""
    if confirmed is None:
        return 0
    count = 0
    for r in _ROW_RANGE:
        for c in _COL_RANGE:
            if int(confirmed.get(r, c)) == COLOR_OJAMA:
                count += 1
    return count


def run_smoke(enable_glow: bool, label: str) -> dict:
    """指定設定でスモーク実行し集計結果を返す。"""
    pipe = RecognitionPipeline.load_default(
        enable_ojama_warning_glow_guard=enable_glow,
    )
    cap = cv2.VideoCapture(str(_CLIP))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # t_start まで skip
    start_frame = int(_T_START * fps)
    end_frame   = int(_T_END   * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 試合 reset のため state machine を初期化
    pipe.reset()

    total_ojama_frames = 0  # おじゃま誤認が 1 件以上あったフレーム数
    total_ojama_cells  = 0  # 誤認セルの延べ数 (1 frame × 1 cell = 1 カウント)
    frame_count = 0

    while True:
        fi = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if fi >= end_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        result = pipe.update(fi, t, frame)
        p1_confirmed = result.p1.confirmed_board
        ojama_cnt = _count_upper_ojama(p1_confirmed)
        if ojama_cnt > 0:
            total_ojama_frames += 1
            total_ojama_cells  += ojama_cnt
        frame_count += 1

    cap.release()
    print(f"[{label}] frames={frame_count} | "
          f"ojama_frames={total_ojama_frames} | ojama_cells={total_ojama_cells}")
    return {
        "label": label,
        "frame_count": frame_count,
        "ojama_frames": total_ojama_frames,
        "ojama_cells": total_ojama_cells,
    }


def main() -> None:
    print(f"スモーク: v89_match01 t={_T_START}-{_T_END}s "
          f"1P row0-4 × col0-5 おじゃま誤認集計")
    print("=" * 60)
    res_off = run_smoke(enable_glow=False, label="OFF (baseline)")
    res_v4  = run_smoke(enable_glow=True,  label="v4 (glow guard)")
    print("=" * 60)
    delta_frames = res_off["ojama_frames"] - res_v4["ojama_frames"]
    delta_cells  = res_off["ojama_cells"]  - res_v4["ojama_cells"]
    print(f"改善: ojama_frames {res_off['ojama_frames']} → {res_v4['ojama_frames']} "
          f"(delta={delta_frames:+d})")
    print(f"改善: ojama_cells  {res_off['ojama_cells']} → {res_v4['ojama_cells']} "
          f"(delta={delta_cells:+d})")
    if delta_frames > 0:
        print("=> v4 でおじゃま誤認 frame が削減 (OK)")
    elif delta_frames == 0:
        print("=> v4 と OFF で変化なし (ガードが発火していない可能性あり)")
    else:
        print("=> v4 でおじゃま誤認 frame が増加 (要調査)")


if __name__ == "__main__":
    main()
