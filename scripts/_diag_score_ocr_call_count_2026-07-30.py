"""ScoreOcr.read_side の呼び出し回数を monkeypatch で直接数える (修正1 専用の軽量計測)。

背景 (2026-07-30):
    共有の _diag_matchtemplate_by_caller_2026-07-30.py は他エージェントが
    並行して編集中 (ui_mask_cells 追加) のため、本スクリプトは依存せず
    自己完結で ScoreOcr.read_side の呼び出し回数だけを数える。
    src/ は一切変更しない (ScoreOcr.read_side をモンキーパッチするのみ)。

実行例 (WSL):
    nice -n 19 ./venv/bin/python -m scripts._diag_score_ocr_call_count_2026-07-30 \
        --video data/frames/video_c60.mp4 --start-sec 1451 --frames 60
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.recognition_pipeline import RecognitionPipeline
from src.score_ocr import ScoreOcr

TARGET_W: int = 1920
TARGET_H: int = 1080


def build_pipeline() -> RecognitionPipeline:
    """本番相当の構成で pipeline を作る。"""
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        force_in_match=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start-sec", type=float, default=1451.0)
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()

    call_count = {"n": 0}
    original = ScoreOcr.read_side

    def _wrapped(self: "ScoreOcr", frame: np.ndarray, side: str):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        return original(self, frame, side)

    ScoreOcr.read_side = _wrapped  # type: ignore[method-assign]

    pipeline = build_pipeline()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(args.start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    done = 0
    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(
                frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA,
            )
        fi = start_frame + i
        pipeline.update(fi, fi / fps, frame)
        done += 1
    cap.release()

    ScoreOcr.read_side = original  # type: ignore[method-assign]

    print(f"処理フレーム数 = {done}")
    print(f"ScoreOcr.read_side 呼び出し回数 = {call_count['n']}")
    if done > 0:
        print(f"1フレームあたり = {call_count['n'] / done:.2f} 回")


if __name__ == "__main__":
    main()
