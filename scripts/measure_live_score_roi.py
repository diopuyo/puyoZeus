"""同じ映像を配信認識経路へOFF/ONで渡し、式・個数入力と所要時間を比較する。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

SECONDS_MS = 1000
FRAME_SIZE = (1920, 1080)
SIDES = ('1P', '2P')


def align_control_band(image: np.ndarray, dy: int) -> np.ndarray:
    """ずれ0対照専用。実映像の得点帯だけを指定ピクセル平行移動する。"""
    if not dy:
        return image
    from src.score_ocr import SCORE_1P_REGION, SCORE_2P_REGION
    from src.score_roi_alignment import SEARCH_RADIUS
    top = min(SCORE_1P_REGION[0], SCORE_2P_REGION[0])-SEARCH_RADIUS
    bottom = max(SCORE_1P_REGION[1], SCORE_2P_REGION[1])+SEARCH_RADIUS
    result = image.copy()
    result[top+dy:bottom+dy] = image[top:bottom]
    return result


def json_value(value: Any) -> Any:
    """盤面を含む返り値を、メモリアドレスに依存せず完全比較する。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, '_grid'):
        return value._grid.tolist()
    if hasattr(value, '_cells'):
        return value._cells
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


def build(enabled: bool) -> tuple:
    """本番採用フラグを読み、任意の式枠フラグだけを変える。"""
    import torch
    from src.recognition_pipeline import RecognitionPipeline
    from src.production_config import recognition_load_default_kwargs, LIVE_ROW1_GUARD_DEFAULT
    from src.b3_observation import ObservationCapture
    torch.manual_seed(0)
    torch.set_num_threads(1)
    config = recognition_load_default_kwargs()
    config.update(enable_score_roi_alignment=enabled,
                  enable_row1_col2_guard=LIVE_ROW1_GUARD_DEFAULT)
    pipe = RecognitionPipeline.load_default(**config)
    # 個数入力の原票まで確認する。予告アイコンは今回の比較対象外。
    observation = ObservationCapture(enable_score_matmul=True, skip_icon_reading=True)
    return pipe, observation


def timed_alignment(pipe: Any, times: dict) -> None:
    """呼出し順を変えず、較正前後の追加処理だけを測る。"""
    alignment = pipe._score_roi_alignment
    original = alignment.observe

    def measured(frame: np.ndarray, time_sec: float) -> None:
        phase = 'locked' if alignment.ready else 'searching'
        begin = perf_counter()
        original(frame, time_sec)
        times[phase].append((perf_counter()-begin)*SECONDS_MS)

    alignment.observe = measured


def summary_ms(values: list[float]) -> dict:
    return dict(n=len(values), mean=float(np.mean(values)) if values else None,
                p95=float(np.percentile(values, 95)) if values else None,
                maximum=max(values) if values else None)


def one_frame(pair: tuple, index: int, stamp: float, image: np.ndarray) -> tuple:
    pipe, observation = pair
    began = perf_counter()
    result = pipe.update(index, stamp, image)
    if pipe._score_roi_alignment is not None:
        observation.ocr._region_offsets = dict(pipe._score_ocr._region_offsets)
    row, _ = observation.update(result, image, index, stamp, (0, 0))
    elapsed = (perf_counter()-began)*SECONDS_MS
    encoded = json.dumps(result, default=json_value, sort_keys=True)
    return encoded, row, elapsed


def compare_frame(pairs: list, index: int, stamp: float, image: np.ndarray) -> list:
    """OFF/ONの実行順を交互にし、CPUキャッシュの順序偏りを減らす。"""
    order = (0, 1) if index % 2 == 0 else (1, 0)
    outputs: list = [None, None]
    for mode in order:
        outputs[mode] = one_frame(pairs[mode], index, stamp, image)
    return outputs


def accumulate(stats: dict, row: dict, pipe: Any) -> None:
    for side, name in zip(row['sides'], SIDES):
        stats['formula_valid'] += int(side['formula']['valid'])
        raw = getattr(pipe, '_formula_last_read_'+name.lower(), None)
        stats['pipeline_valid'] += int(bool(raw and raw.valid))
        step = side['formula_step']
        if step is not None:
            stats['steps'] += 1
            stats['erased'] += step['erased']
        stats['max_ncc'] = max(stats['max_ncc'], side['formula']['mult_ncc'])


def run(args: argparse.Namespace) -> dict:
    from src.phase_j.dshow_color import correct_601_as_709
    pairs = [build(False), build(True)]
    timings: dict = dict(searching=[], locked=[])
    timed_alignment(pairs[1][0], timings)
    processing: list[list[float]] = [[], []]
    stats = [dict(formula_valid=0, pipeline_valid=0, steps=0, erased=0, max_ncc=0.) for _ in pairs]
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise OSError(f'動画を開けません: {args.video}')
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start*SECONDS_MS)
    result = dict(frames=0, result_mismatches=0, input_mismatches=0, locked_at=None)
    begin = perf_counter()
    try:
        for index in range(round(args.duration*fps)):
            ok, frame = cap.read()
            if not ok:
                break
            frame = correct_601_as_709(cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA))
            frame = align_control_band(frame, args.score_band_dy)
            outputs = compare_frame(pairs, index, args.start+index/fps, frame)
            result['result_mismatches'] += int(outputs[0][0] != outputs[1][0])
            result['input_mismatches'] += int(outputs[0][1] != outputs[1][1])
            for pair, total, durations, (_, row, elapsed) in zip(pairs, stats, processing, outputs):
                accumulate(total, row, pair[0])
                durations.append(elapsed)
            result['frames'] += 1
            if result['locked_at'] is None and pairs[1][0]._score_roi_alignment.ready:
                result['locked_at'] = index/fps
            if index % round(fps*10) == 0:
                print(index, '経過秒', round(perf_counter()-begin, 1), flush=True)
    finally:
        cap.release()
        for _, observation in pairs:
            observation.close()
    for total in stats:
        total['read_percent'] = total['formula_valid']/(len(SIDES)*result['frames'])*100
    return dict(result, fps=fps, stats=stats, score_band_dy=args.score_band_dy,
                row1_guard=pairs[0][0]._enable_row1_col2_guard,
                processing_ms=[summary_ms(v) for v in processing],
                status=pairs[1][0]._score_roi_alignment.status(),
                alignment_ms={k: summary_ms(v) for k, v in timings.items()}, wall_sec=perf_counter()-begin)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--duration', type=float, default=180.)
    parser.add_argument('--score-band-dy', type=int, choices=(0, 1), default=0,
                        help='対照試験専用: 得点帯だけを1px下へ揃える（既定0、無加工）')
    args = parser.parse_args()
    args.video, args.output = args.video.resolve(), args.output.resolve()
    os.chdir(args.asset_root)
    result = run(args)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
