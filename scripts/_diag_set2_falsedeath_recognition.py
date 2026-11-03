"""元の収集条件で認識を再生し、確定盤面の書込み元を観測する。"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path('/mnt/d/puyo_analyzer/wt_evalset')
OUT = ROOT / 'logs/eval_set/set2_falsedeath/recognition'
LOOKBACK = 12.0
TAIL = 1.0
WARMUP = 30.0
STRIDE = 2


def signature(side: Any) -> dict:
    """補助観測を付ける前の認識出力を比較する。"""
    board = side.confirmed_board
    event = side.chain_event
    return dict(state=side.state.name, board=None if board is None else board._grid.tolist(),
                score=side.score, next=side.next_pair, dnext=side.dnext_pair,
                event=None if event is None else dict(trigger=event.trigger_sec,
                    mechanism=event.mechanism, count=event.chain_count, score=event.total_score))


def direct(part: str, start: float, end: float, stamps: list[float]) -> None:
    """描画・評価を省き、元の認識呼出しと全ネイティブフレーム観測だけを再現する。"""
    from src.recognition_pipeline import RecognitionPipeline
    from src.exchange_event_record import read_records
    from scripts._diag_set2_falsedeath_extract import VIDEO
    config = json.loads((ROOT / 'logs/eval_set/set2/collect/capture' / part / 'effective_config.json').read_text())
    pipe = RecognitionPipeline.load_default(**config)
    capture = cv2.VideoCapture(str(VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS)
    first = int(max(0, start - WARMUP) * fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    expected = {round(r['args'][3] * fps): r['args'][0]
                for r in read_records(ROOT / 'logs/eval_set/set2/collect/records' / f'{part}.jsonl.gz')
                if r['kind'] == 'update' and r['args'][3] <= end}
    compared, mismatches, target_rows = 0, [], []
    for index in range(first, int(end * fps)):
        ok, frame = capture.read()
        assert ok
        stamp = index / fps
        pipe.observe_placement_frame(index, stamp, frame)
        if (index - first) % STRIDE:
            continue
        actual = pipe.update(index, stamp, frame)
        if index in expected:
            compared += 1
            lhs = [signature(expected[index].p1), signature(expected[index].p2)]
            rhs = [signature(actual.p1), signature(actual.p2)]
            if lhs != rhs:
                mismatches.append(stamp)
            if any(abs(stamp - t) <= TAIL for t in stamps):
                target_rows.append(dict(t_sec=stamp, identical=lhs == rhs, expected=lhs, actual=rhs))
        if index % round(fps * 10) == first % STRIDE:
            print(f'{part} 認識 {stamp:.3f}s', flush=True)
    capture.release()
    result = dict(compared=compared, mismatches=mismatches, target_rows=target_rows)
    (OUT / f'{part}_identity.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')


def install_trace(stamps: list[float], path: Path) -> None:
    """対象直前だけ行単位で盤面変化を記録する。計算入力には介入しない。"""
    from src.recognition_pipeline import RecognitionPipeline
    original = RecognitionPipeline._step_side
    signature = inspect.signature(original)
    stream = path.open('w', encoding='utf-8')

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(self, *args, **kwargs).arguments
        stamp = bound['time_sec']
        if not any(t - LOOKBACK <= stamp <= t + TAIL for t in stamps):
            return original(self, *args, **kwargs)
        previous: dict = {'grid': None, 'line': None}

        def trace(frame: Any, event: str, arg: Any) -> Any:
            if frame.f_code is not original.__code__:
                return None
            ctx = frame.f_locals.get('ctx')
            if ctx is not None and ctx.confirmed_board is not None:
                grid = ctx.confirmed_board._grid
                if previous['grid'] is None or not np.array_equal(previous['grid'], grid):
                    row = dict(t_sec=stamp, side=bound['side'], previous_line=previous['line'],
                               next_line=frame.f_lineno, state=ctx.state.name,
                               before=previous['grid'].tolist() if previous['grid'] is not None else None,
                               after=grid.tolist())
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    previous['grid'] = grid.copy()
            previous['line'] = frame.f_lineno
            return trace

        sys.settrace(trace)
        try:
            return original(self, *args, **kwargs)
        finally:
            sys.settrace(None)

    RecognitionPipeline._step_side = wrapped


def main() -> None:
    """元のパート開始・ウォームアップを保ち、最終対象直後で止める。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('part')
    parser.add_argument('--direct', action='store_true')
    args = parser.parse_args()
    cases = json.loads((OUT.parent / 'cases.json').read_text())
    stamps = [c['first_sec'] for c in cases if c['label']['part'] == args.part]
    command = json.loads((ROOT / 'logs/eval_set/set2/collect/capture' / args.part / 'COMMAND.json').read_text())['args']
    start = float(command[command.index('--start-sec') + 1])
    OUT.mkdir(parents=True, exist_ok=True)
    install_trace(stamps, OUT / f'{args.part}_writes.jsonl')
    if args.direct:
        direct(args.part, start, max(stamps) + TAIL, stamps)
        return
    sys.argv = ['collect', '--part', args.part, '--start-sec', str(start),
                '--end-sec', str(max(stamps) + TAIL), '--out', str(OUT / args.part)]
    from scripts.collect_eval_set_20261001 import main as collect
    collect()


if __name__ == '__main__':
    main()
