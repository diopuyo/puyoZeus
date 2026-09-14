"""原 collector caller と v11 全経路。拒否前の実値も必ず保存。"""
from __future__ import annotations
from contextlib import ExitStack
from copy import deepcopy
import traceback
from typing import Any
import artificial_inputs as I
import observations as O


def loop(q: Any, target: Any, m: Any, state: Any, real: Any, fixture: Any,
         binding: Any, factory: Any, data: dict[str, Any]) -> None:
    pipe, rec = real[:2]
    collector, pixels = m.loop.collector_loop(), binding['provider']
    collector.RecognitionPipeline = type(pipe)
    cap = m.smoke.SyntheticCapture(real, fixture, collector.cv2, pixels.o.np)
    originals = type(pipe).update, collector.cv2.resize, type(pixels).__call__, pipe._reader.read_both_boards
    clock = {'frame': I.FRAMES[0]}
    try:
        with ExitStack() as stack:
            data['flags'], data['control'] = O.flags(pipe), O.control()
            data['supplied'] = I.inputs(q, stack, pipe, pixels, factory.types.parts.O, clock)
            reader = I.score_input(stack, pipe, clock)
            data['ocr'] = reader.rows
            data['bodies'] = q.profiling(stack, factory, m.recording)
            data['origin_bodies'] = O.profiling(stack, pipe, factory, clock)
            sink = m.recording.install(stack, state, collector, fixture.subject.base)
            raw = m.raw.install(stack, collector, pixels, rec.emit)
            for frame in I.FRAMES:
                clock['frame'], cap.position = frame, frame
                I.paint_score(cap, frame)
                try:
                    collector.collect_lean(cap, pipe, frame, 1, q.K.STRIDE, q.K.FPS)
                    data['completed'].append(frame)
                except BaseException as error:
                    data['primary'] = dict(frame=frame, type=type(error).__name__, message=str(error),
                                           traceback=traceback.format_exc())
                    raise
                finally:
                    data['trace'].append(O.snapshot(target, state, factory, pipe, frame))
    finally:
        data['raw_input_references_restored'] = originals == (type(pipe).update, collector.cv2.resize,
            type(pixels).__call__, pipe._reader.read_both_boards)
        data['score_input_references_restored'] = all(getattr(pipe, n) is None
            for n in ('_score_ocr', '_score_tracker_1p', '_score_tracker_2p'))
        if 'raw' in locals():
            data['raw'] = dict(closed=raw.closed, error=raw.error)
            data['sink'] = dict(closed=sink.closed, errors=sink.errors, rows=sink.rows)


def drive(q: Any, target: Any, m: Any, state: Any, real: Any, fixture: Any,
          binding: Any, factory: Any) -> Any:
    pipe = real[0]
    counter = {s: (getattr(pipe, '_tsumo_count_'+s), deepcopy(getattr(pipe, '_tsumo_count_'+s)))
               for s in ('1p', '2p')}
    data: dict[str, Any] = dict(trace=[], completed=[], primary=None)
    try:
        loop(q, target, m, state, real, fixture, binding, factory, data)
        raise AssertionError('firing_origin_rejection_not_observed')
    finally:
        data['native_counter_unchanged'] = all(getattr(pipe, '_tsumo_count_'+s) is ref and ref == old
            for s, (ref, old) in counter.items())
        data['adoptions'] = factory.baseline_adoption.adoptions
        receiver = state['postcommit_current_receiver']
        data['publication'] = dict(issued=receiver.issued, released=receiver.released, errors=receiver.errors)
        data['first_move_final'] = pipe._first_move_sec_1p
        data['physical_certified'], data['collector_append_tested'] = False, False
        q.K.write(state['output']/'DIAGNOSTIC.json', fixture.subject.base.json_value(data))
