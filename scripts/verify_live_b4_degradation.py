"""B3の611セルを分解し、NV12再圧縮2水準と既存HSV継続較正を比較する。"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import cv2
import numpy as np

from scripts.verify_live_degradation_20260928 import (
    START, END, WARMUP, FPS, BOARD_SHAPE, DEFAULT_VIDEO, transcode, compare_cells)
from scripts.run_live_pipeline_20260928 import save_json
from src.phase_j.live_source import VideoFileSource

OUTPUT = Path('logs/live_b4_degradation')
CONFIG = Path('logs/live_b3_realtime_off/recognition_config.json')
QUALITY_LEVELS = (23, 35)
COLORS = (0, 1, 2, 3, 4, 5, 9, 10)
HUE_PERIOD = 180
HUE_HALF_PERIOD = HUE_PERIOD // 2


def frame_details(frame: np.ndarray, result: Any, pipe: Any) -> dict[str, np.ndarray]:
    """確定値と瞬間観測を区別し、可視セル中央パッチのHSVを保存する。"""
    from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
    grid = np.full(BOARD_SHAPE, -1, dtype=np.int16)
    hsv = np.full((*BOARD_SHAPE, 3), np.nan, dtype=np.float32)
    hsv_colors = np.full(BOARD_SHAPE, -1, dtype=np.int16)
    states, stable, raw = [], [], []
    for index, (side, region) in enumerate(zip((result.p1, result.p2),
                                              (DEFAULT_P1_REGION, DEFAULT_P2_REGION))):
        available = side.state.name == 'STABLE' and side.confirmed_board is not None
        stable.append(available)
        states.append(side.state.name)
        if available:
            grid[index] = side.confirmed_board._grid
        raw.append(side.cnn_board._grid.copy())
        _, hsv_grid = pipe._reader._classifier.predict_proba_and_hsv_grid(frame, region)
        hsv_colors[index, 1:] = hsv_grid
        for row in range(1, BOARD_SHAPE[1]):
            for col in range(BOARD_SHAPE[2]):
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                patch = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
                hsv[index, row, col] = np.median(patch, axis=(0, 1))
    return dict(boards=grid, stable=np.array(stable), states=np.array(states),
                raw=np.array(raw), hsv=hsv, hsv_colors=hsv_colors)


def recognize(video: Path, config: dict, clipped: bool, refresh: bool = False) -> dict:
    """B3と同じ乱数・原動画時刻を使い、較正スイッチだけを変更する。"""
    import torch
    from src.recognition_pipeline import RecognitionPipeline
    from scripts.run_e3_exchange_eval_20260926 import SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    pipe = RecognitionPipeline.load_default(**dict(config, enable_online_hsv_refresh=refresh))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    first = round((START-WARMUP)*fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0 if clipped else first)
    source = VideoFileSource(cap, fps, first, round(END*fps), round(fps/FPS))
    rows, injections = [], []
    previous = set()
    try:
        for frame in source:
            result = pipe.update(frame.index, frame.media_sec, frame.image)
            colors = pipe._online_hsv_injected_colors
            if colors != previous:
                injections.append(dict(t_sec=frame.media_sec, colors=sorted(colors)))
                previous = set(colors)
            if frame.media_sec >= START:
                rows.append(frame_details(frame.image, result, pipe))
    finally:
        cap.release()
    arrays = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    arrays['calibration'] = dict(injections=injections, refresh=refresh,
        sample_counts=pipe._online_hsv.get_sample_counts(),
        ranges=pipe._online_hsv.get_per_video_ranges())
    return arrays


def counts(values: list) -> dict[str, int]:
    return dict(sorted(Counter(map(str, values)).items()))


def breakdown(left: dict, right: dict, mask: np.ndarray | None = None) -> dict:
    """同時STABLEの差分を混同・位置・状態遷移・瞬間色観測に分解する。"""
    if mask is None:
        mask = left['stable'] & right['stable']
    diff = (left['boards'] != right['boards']) & mask[:, :, None, None]
    indices = np.argwhere(diff)
    confusion = np.zeros((len(COLORS), len(COLORS)), dtype=int)
    pairs, transitions, hsv_pairs = [], [], []
    for frame, side, row, col in indices:
        a, b = int(left['boards'][frame, side, row, col]), int(right['boards'][frame, side, row, col])
        confusion[COLORS.index(a), COLORS.index(b)] += 1
        pairs.append(f"{left['states'][frame, 0]}/{left['states'][frame, 1]} -> "
                     f"{right['states'][frame, 0]}/{right['states'][frame, 1]}")
        transitions.append(f"{left['states'][max(0, frame-1), side]} -> STABLE / "
                           f"{right['states'][max(0, frame-1), side]} -> STABLE")
        hsv_pairs.append(f"{left['hsv_colors'][frame, side, row, col]} -> "
                         f"{right['hsv_colors'][frame, side, row, col]}")
    delta = right['hsv'] - left['hsv']
    delta[..., 0] = (delta[..., 0] + HUE_HALF_PERIOD) % HUE_PERIOD - HUE_HALF_PERIOD
    visible = diff.copy()
    visible[:, :, 0] = False
    raw_diff = left['raw'] != right['raw']
    hsv_diff = left['hsv_colors'] != right['hsv_colors']
    return dict(cells=int(mask.sum()*np.prod(BOARD_SHAPE[1:])), different_cells=int(diff.sum()),
        colors=COLORS, confusion=confusion.tolist(), by_side_row_col=diff.sum(axis=0).tolist(),
        frame_types=counts(pairs), previous_states=counts(transitions), hsv_confusion=counts(hsv_pairs),
        raw_disagreements_at_differences=int((raw_diff & diff).sum()),
        hsv_disagreements_at_visible_differences=int((hsv_diff & visible).sum()),
        visible_differences=int(visible.sum()), hidden_differences=int(diff[:, :, 0].sum()),
        absolute_hsv_delta_percentiles=np.percentile(np.abs(delta[visible]), (50, 95, 99), axis=0).tolist()
        if visible.any() else [], indices=indices.tolist())


def save_recognition(name: str, result: dict) -> None:
    np.savez_compressed(OUTPUT / f'{name}.npz', **{k: v for k, v in result.items() if k != 'calibration'})
    save_json(OUTPUT / f'{name}_calibration.json', result['calibration'])


def compare(left: dict, right: dict) -> dict:
    report = compare_cells((left['boards'], left['stable']), (right['boards'], right['stable']))
    report['breakdown'] = breakdown(left, right)
    return report


def quality_run(crf: int, original: dict, config: dict) -> dict:
    """一水準ずつ処理し、動画を破棄して盤面・較正・変換hashだけ保持する。"""
    video = OUTPUT / f'nv12_crf{crf}.mp4'
    command = transcode(DEFAULT_VIDEO, video, crf)
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    try:
        baseline = recognize(video, config, True)
        save_recognition(f'crf{crf}_baseline', baseline)
        calibrated = recognize(video, config, True, refresh=True)
        save_recognition(f'crf{crf}_calibrated', calibrated)
    finally:
        video.unlink()
    common = original['stable'] & baseline['stable'] & calibrated['stable']
    report = dict(crf=crf, command=command, sha256=digest,
        baseline=compare(original, baseline), calibrated=compare(original, calibrated),
        common_baseline=breakdown(original, baseline, common),
        common_calibrated=breakdown(original, calibrated, common))
    if crf == QUALITY_LEVELS[0]:
        with np.load('logs/live_b3_degradation/boards.npz') as saved:
            report['b3_reproduction'] = dict(
                original_equal=bool(np.array_equal(saved['original'], original['boards'])),
                degraded_equal=bool(np.array_equal(saved['degraded'], baseline['boards'])))
    save_json(OUTPUT / f'crf{crf}.json', report)
    return report


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    original = recognize(DEFAULT_VIDEO, config, False)
    save_recognition('original', original)
    for crf in QUALITY_LEVELS:
        quality_run(crf, original, config)


if __name__ == '__main__':
    main()
