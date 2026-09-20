"""G3の連続入力契約と、認識評価から分離した小規模デコード検証。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any, Callable

from scripts.g3_agent_review import ROOT, VERIFY, read, save
from src.event_run_v1 import SourceVideoSpec
from src.fps_normalize import resolve_normalize_fps_30_stride

PREREG = ROOT / 'docs/agent_coordination/G3_VIDEO_VALIDATION_PREREG_2026-09-07.json'
SOURCE_ORDER = ('video_38', 'video_39', 'video_c74', 'video_c50', 'video_c80', 'video_c138')
PROBE_FRAMES = 4
FORMAT = 'g3-decode-only-probe/1'
POSITION_PROPERTY = 1  # OpenCV CAP_PROP_POS_FRAMES


def validate(source: SourceVideoSpec) -> None:
    """未確定値・途中開始を拒否し、登録された半開区間を検査する。"""
    integers = (source.width, source.height, source.frame_count,
                source.time_base_numerator, source.time_base_denominator,
                source.processing_start_frame, source.processing_end_frame_exclusive)
    if any(type(value) is not int for value in integers):
        raise ValueError('source_integer_required')
    if min(integers[:5]) <= 0 or source.processing_start_frame != 0:
        raise ValueError('source_positive_timebase_and_frame_zero_required')
    if not 0 < source.processing_end_frame_exclusive <= source.frame_count:
        raise ValueError('invalid_exclusive_end')


def contracts(prereg: Path = PREREG) -> list[SourceVideoSpec]:
    """元動画の総数と評価prefixを混同せず、既存事前登録を読み込む。"""
    items = read(prereg)['sources']
    if tuple(item['source_video_id'] for item in items) != SOURCE_ORDER:
        raise ValueError('source_order_changed')
    result = []
    for item in items:
        prefix = item['required_continuous_processing']
        if prefix['seek_between_windows'] is not False:
            raise ValueError('continuous_history_required')
        source = replace(SourceVideoSpec(**item['source']),
                         processing_start_frame=prefix['start_frame'],
                         processing_end_frame_exclusive=prefix['end_frame_exclusive'])
        validate(source)
        result.append(source)
    return result


def frame_time(source: SourceVideoSpec, frame: int) -> Fraction:
    """frameに直接対応する有理時刻。strideで時刻を積算しない。"""
    validate(source)
    if type(frame) is not int or not 0 <= frame < source.processing_end_frame_exclusive:
        raise ValueError('frame_outside_contract')
    return Fraction(frame * source.time_base_numerator, source.time_base_denominator)


def stride_for(source: SourceVideoSpec) -> int:
    """既存の丸め規則をcanonical fpsへ適用。59.94fpsは実効29.97fps。"""
    validate(source)
    return resolve_normalize_fps_30_stride(
        source.time_base_denominator / source.time_base_numerator)


def consume(cap: Any, source: SourceVideoSpec, end: int,
            visitor: Callable[[int, Fraction, Any], None], state: dict[str, Any]) -> None:
    """全frameを順次読み、既存strideのframeのみ渡す。早期EOFは必ず失敗。"""
    validate(source)
    if type(end) is not int or not 0 < end <= source.processing_end_frame_exclusive:
        raise ValueError('end_outside_contract')
    stride = stride_for(source)
    if cap.get(POSITION_PROPERTY) != 0:
        raise ValueError('capture_not_at_frame_zero')
    for frame in range(end):
        state['attempted_frame'] = frame
        ok, pixels = cap.read()
        if not ok or pixels is None:
            raise RuntimeError(f'early_eof:{frame}')
        if cap.get(POSITION_PROPERTY) != frame + 1:
            raise RuntimeError(f'decode_position_mismatch:{frame}')
        if tuple(pixels.shape[:2]) != (source.height, source.width):
            raise RuntimeError(f'decode_shape_mismatch:{frame}')
        state.update(decoded_count=frame + 1, last_decoded_frame=frame)
        if frame % stride == 0:
            visitor(frame, frame_time(source, frame), pixels)
            state['delivered_count'] += 1


def close_resources(cap: Any, stream: Any) -> list[dict[str, str]]:
    """片方の終了失敗でも他方を解放し、各例外を落とさない。"""
    errors = []
    actions = [('flush', stream.flush), ('fsync', lambda: os.fsync(stream.fileno())),
               ('close', stream.close), ('release', cap.release)]
    for name, action in actions:
        try:
            action()
        except BaseException as error:
            errors.append(dict(stage=name, type=type(error).__name__, message=str(error)))
    return errors


def run_probe(source: SourceVideoSpec, output: Path,
              capture_factory: Callable[[str], Any]) -> dict[str, Any]:
    """小デコードだけを実行する。認識器・GT・G3合格への入口は提供しない。"""
    validate(source)
    if not output.resolve().is_relative_to(VERIFY.resolve()):
        raise ValueError('output_must_be_on_verify_drive')
    output.mkdir(parents=True, exist_ok=False)
    end = min(PROBE_FRAMES, source.processing_end_frame_exclusive)
    state: dict[str, Any] = dict(format=FORMAT, mode='decode_only', quality_pass=False,
        source=asdict(source), probe_end_exclusive=end, decoded_count=0,
        delivered_count=0, attempted_frame=None, last_decoded_frame=None,
        status='FAILED', error=None, cleanup_errors=[], source_sha_rechecked=False)
    save(output / 'PROBE_CONTRACT.json', state)
    started, cap, stream = time.monotonic(), None, None
    try:
        stream = (output / 'DECODE_ONLY_FRAMES.jsonl').open('x', encoding='utf-8')
        cap = capture_factory(str(ROOT / source.source_video_path))
        if not cap.isOpened():
            raise RuntimeError('capture_open_failed')
        def record(frame: int, stamp: Fraction, pixels: Any) -> None:
            row = dict(format=FORMAT, frame=frame, time_num=stamp.numerator,
                       time_den=stamp.denominator, shape=list(pixels.shape),
                       decoded_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
            stream.write(json.dumps(row) + '\n')
        consume(cap, source, end, record, state)
        state['status'] = 'DECODE_ONLY_PASS'
    except BaseException as error:
        state['error'] = dict(type=type(error).__name__, message=str(error),
                             traceback=traceback.format_exc())
    finally:
        state['cleanup_errors'] = cleanup(cap, stream)
        if state['cleanup_errors']:
            state['status'] = 'FAILED'
        state['seconds'] = time.monotonic() - started
        state['release_succeeded'] = cap is not None and not any(
            error['stage'] == 'release' for error in state['cleanup_errors'])
        save(output / 'PROBE_RESULT.json', state)
    return state


def cleanup(cap: Any, stream: Any) -> list[dict[str, str]]:
    """生成前失敗でも、生成できた資源は閉じる。"""
    if cap is not None and stream is not None:
        return close_resources(cap, stream)
    errors = []
    for name, resource in [('close', stream), ('release', cap)]:
        if resource is None:
            continue
        try:
            getattr(resource, name)()
        except BaseException as error:
            errors.append(dict(stage=name, type=type(error).__name__, message=str(error)))
    return errors


def main() -> int:
    """6本を独立に診断し、結果と未実行理由をDへ保存する。"""
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(VERIFY.resolve()):
        parser.error('output_must_be_on_verify_drive')
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for source in contracts():
        if results and results[-1]['status'] == 'STOP_REMAINING':
            results.append(dict(source=source.source_video_id, status='STOP_REMAINING',
                                reason='prior_resource_or_interrupt_failure'))
            continue
        result = run_probe(source, args.output / source.source_video_id, cv2.VideoCapture)
        stop = result['cleanup_errors'] or (result['error'] or {}).get('type') in {
            'KeyboardInterrupt', 'SystemExit', 'MemoryError'}
        results.append(dict(source=source.source_video_id,
                            status='STOP_REMAINING' if stop else result['status'],
                            error=result['error'], cleanup_errors=result['cleanup_errors']))
    save(args.output / 'PROBE_INDEX.json', dict(results=results, quality_pass=False,
         recognition_started=False, g3_status='INCOMPLETE'))
    print(json.dumps(results))
    return int(any(row['status'] != 'DECODE_ONLY_PASS' for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
