"""動画 frame の高速読み込みヘルパ (Z-3C: cap.read 並列化)。

cv2.VideoCapture.set(POS_MSEC) + read() を 0.1s 刻みで呼ぶと、内部的に
近傍 keyframe からの seek + decode が発生し、各 frame で 50-100ms かかる。
連続 frame であれば順次 read() の方が速い (decode 連続のため)。

本モジュールは:
    1. 連続 frame batch reader: 開始時刻から時刻リストを順次読む
    2. background thread での先読み: 並列 thread が常に N frame 先読み

時刻リストが等間隔 (例: 0.1s 毎) かつ動画が前から後への線形再生なら、
連続 read() で十分高速 (cap.set 不要、frame skip だけ)。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameSample:
    """1 frame 分の読み込み結果。"""
    t_sec: float
    frame: np.ndarray  # 1080p に正規化済 BGR


def read_frames_sequential(
    video_path: str,
    times_sec: list[float],
    target_size: tuple[int, int] = (1920, 1080),
) -> list[FrameSample | None]:
    """指定時刻リストの frame を順次読み込み (cap.set 最小化)。

    times_sec は昇順想定。連続 frame なら 1 度の cap.set で開始位置を
    決め、後は次の frame まで read() を grab() で skip する。

    Args:
        video_path: 動画パス。
        times_sec: 読み込みたい時刻のリスト (昇順想定)。
        target_size: 出力 frame サイズ (1920x1080 想定)。

    Returns:
        各時刻に対応する FrameSample (read 失敗なら None)。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [None] * len(times_sec)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_w, target_h = target_size

    out: list[FrameSample | None] = [None] * len(times_sec)
    if not times_sec:
        cap.release()
        return out

    # 開始位置に seek
    start_t = times_sec[0]
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_t) * 1000)

    cur_frame_idx = int(round(start_t * fps))
    for i, t in enumerate(times_sec):
        target_idx = int(round(t * fps))
        # cur_frame_idx が target に達するまで grab() で skip
        skip = target_idx - cur_frame_idx
        if skip < 0:
            # 後戻り (時刻順序が逆) は seek
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
            cur_frame_idx = target_idx
        elif skip > 0:
            for _ in range(skip):
                if not cap.grab():
                    break
                cur_frame_idx += 1
        ok, frame = cap.read()
        if not ok or frame is None:
            cur_frame_idx += 1
            continue
        cur_frame_idx += 1
        if frame.shape[1] != target_w or frame.shape[0] != target_h:
            frame = cv2.resize(
                frame, (target_w, target_h),
                interpolation=cv2.INTER_AREA,
            )
        out[i] = FrameSample(t_sec=t, frame=frame)
    cap.release()
    return out


__all__ = [
    "FrameSample",
    "read_frames_sequential",
]
