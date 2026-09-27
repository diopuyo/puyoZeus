"""NV12 4:2:0往復とH.264再圧縮の認識差を、同一時刻・同一設定で数える。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np

from scripts.measure_realtime_breakdown_20260928 import DEFAULT_VIDEO
from scripts.run_live_pipeline_20260928 import save_json
from src.phase_j.live_source import VideoFileSource

START, END, WARMUP = 2600.0, 2640.0, 1.0
FPS = 30
CRF = 23
ENCODE_THREADS = 2
BOARD_SHAPE = (2, 13, 6)


def transcode(video: Path, output: Path) -> list[str]:
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-ss', str(START-WARMUP),
        '-i', str(video), '-t', str(END-START+WARMUP), '-an', '-vf',
        'scale=1920:1080,format=nv12,format=yuv420p', '-c:v', 'libx264',
        '-crf', str(CRF), '-preset', 'veryfast', '-threads', str(ENCODE_THREADS), str(output)]
    with output.with_suffix('.ffmpeg.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return command


def recognize_boards(video: Path, config: dict[str, Any], clipped: bool) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from src.recognition_pipeline import RecognitionPipeline
    from scripts.run_e3_exchange_eval_20260926 import SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    pipe = RecognitionPipeline.load_default(**config)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    first = round((START-WARMUP)*fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0 if clipped else first)
    source = VideoFileSource(cap, fps, first, round(END*fps), round(fps/FPS))
    boards, stable = [], []
    try:
        for frame in source:
            result = pipe.update(frame.index, frame.media_sec, frame.image)
            if frame.media_sec < START:
                continue
            grid = np.full(BOARD_SHAPE, -1, dtype=np.int16)
            states = []
            for index, side in enumerate((result.p1, result.p2)):
                available = side.state.name == 'STABLE' and side.confirmed_board is not None
                states.append(available)
                if available:
                    grid[index] = side.confirmed_board._grid
            boards.append(grid)
            stable.append(states)
    finally:
        cap.release()
    return np.array(boards), np.array(stable)


def compare_cells(original: tuple[np.ndarray, np.ndarray],
                  degraded: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    boards_a, stable_a = original
    boards_b, stable_b = degraded
    if boards_a.shape != boards_b.shape:
        raise ValueError('比較動画のフレーム対応が一致しません')
    mask = stable_a & stable_b
    differences = (boards_a != boards_b) & mask[:, :, None, None]
    indices = np.argwhere(differences)
    first = indices[0].tolist() if len(indices) else None
    return dict(frames=len(boards_a), compared_side_frames=int(mask.sum()),
        cells=int(mask.sum()*np.prod(BOARD_SHAPE[1:])), different_cells=int(differences.sum()),
        stable_availability_mismatches=int((stable_a != stable_b).sum()),
        original_stable_side_frames=int(stable_a.sum()), degraded_stable_side_frames=int(stable_b.sum()),
        first_difference=first,
        definition='両方STABLEかつconfirmed_boardありの同時刻同side。隠し段も含む13x6。正解率ではない。')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=DEFAULT_VIDEO)
    parser.add_argument('--recognition-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('logs/live_b3_degradation'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    video = args.output / 'nv12_recompressed.mp4'
    command = transcode(args.video, video)
    config = json.loads(args.recognition_config.read_text())
    original = recognize_boards(args.video, config, False)
    degraded = recognize_boards(video, config, True)
    np.savez_compressed(args.output / 'boards.npz', original=original[0], degraded=degraded[0],
                        original_stable=original[1], degraded_stable=degraded[1])
    report = compare_cells(original, degraded)
    report.update(command=command, encoded_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                  recognition_config=config)
    save_json(args.output / 'comparison.json', report)
    # 生成した動画だけを削除する。元動画は読み取り専用のまま残す。
    video.unlink()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
