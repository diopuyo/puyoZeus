"""人工視覚信号を原full collectへ通し、境界確定と会計原票を同時保存。"""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
import pytest
import test_capture as T
import save_proof as P

ROOT = Path(__file__).resolve().parent
RISE, LAST = 60, 92


def main() -> None:
    before, start = P.sources(), time.monotonic()
    c, b = T.real.__wrapped__()
    output = ROOT / 'full_boundary_v2'
    output.mkdir()
    state, pipe = {'output': output}, T.T.pipeline(c)
    update = pipe.update
    def signal(frame_idx: int, time_sec: float, frame: object) -> object:
        result = update(frame_idx, time_sec, frame)
        result.is_match_active = frame_idx >= RISE
        return result
    pipe.update = signal
    with pytest.MonkeyPatch.context() as patch, ExitStack() as stack:
        patch.setattr(b.O, 'FIRST', 0)
        patch.setattr(b.O, 'LAST', LAST)
        b.install(stack, c, None, state, enabled=True)
        tail = T.T.M.install(stack, state)
        loop = T.U.build(c, b.O.SOURCE, overrides=T.L.settings())
        stack.callback(loop.generator.close)
        original = loop.collect_lean
        loop.RecognitionPipeline = type(pipe)
        capture = T.C.install(stack, loop, tail, {'source': 'artificial-boundary', 'run': 'full_boundary_v2'})
        cap = T.N(read=lambda: (True, T.T.np.zeros((1080, 1920, 3), dtype=T.T.np.uint8)))
        for frame in range(0, LAST + T.C.STRIDE, T.C.STRIDE):
            loop.collect_lean(cap, pipe, frame, 1, T.C.STRIDE, T.C.FPS)
        saved = capture.snapshot()
    assert capture.closed and loop.collect_lean is original
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(b.O, 'FIRST', 0)
        patch.setattr(b.O, 'LAST', LAST)
        b.finish(state)
        b.verify(output)
    verify(saved, before, start, output)


def verify(saved: dict, before: dict, start: float, output: Path) -> None:
    accepted = [e for e in saved['boundary_events'] if e['list'] == 'advance_times']
    assert [(e['raw_value'], e['available_frame']) for e in accepted] == [(1.0, 90)]
    rows = saved['accounting']['rows']
    assert any(row['frame_idx'] == 90 and row['formal_boundary'] for row in rows)
    assert before == P.sources()
    saved.update(source_unchanged=True, sources=before, actual_video=False,
                 artificial_pipeline=True, seconds=time.monotonic() - start,
                 runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (output / 'CAPTURE.json').open('x', encoding='utf-8') as stream:
        json.dump(saved, stream, ensure_ascii=False, indent=2)
    print('原full collectの境界確定90・発生60・会計formal同90: PASS限定')


if __name__ == '__main__':
    main()
