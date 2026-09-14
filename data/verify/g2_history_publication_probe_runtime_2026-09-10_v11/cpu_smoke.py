"""原full constructor/loopとraw外側を一回接続。空盤面/ScoreZeroは人工。"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace, FunctionType
from typing import Any
import assembly_publication_probe as A
import common as K
import goals as G

FRAMES = (29052, 29054, 29056, 29058)
ARTIFICIAL_OLD_PAIR = (1,1)
RAW = K.VERIFY / 'g2_directional_raw_input_2026-09-09_v3/bridge.py'
SMOKE = K.VERIFY / 'g2_directional_raw_input_independent_2026-09-09_v1/smoke.py'
LOOP = K.VERIFY / 'g2_directional_raw_input_2026-09-09_v1/test_bridge.py'


def loaded(stack: Any) -> Any:
    smoke = K.load('_combined_original_loop_helper', SMOKE, stack)
    raw = K.load('_combined_raw_v3', RAW, stack)
    loop = smoke.load('_combined_loop_fixture', LOOP, {'bridge': raw})
    stack.callback(sys.modules.pop, '_combined_loop_fixture', None)
    cpu = SimpleNamespace(**(vars(K) | {'FRAMES': FRAMES, 'FIRST': FRAMES[0], 'END': FRAMES[-1]+K.STRIDE}))
    cpu.bounds = FunctionType(K.bounds.__code__, dict(vars(K), FRAMES=FRAMES, FIRST=cpu.FIRST, END=cpu.END))
    recording = smoke.load('_combined_CPU_recording', K.ROOT/'recording.py', {'common': cpu})
    stack.callback(sys.modules.pop, '_combined_CPU_recording', None)
    return SimpleNamespace(smoke=smoke, raw=raw, loop=loop, recording=recording, cpu=cpu)


def tracked(factory: Any, rows: list[Any], previous: Any) -> Any:
    code = type(factory.provider).view.__code__
    def profile(frame: Any, event: str, value: Any) -> None:
        if previous is not None:
            previous(frame, event, value)
        if frame.f_code is code and event == 'return' and frame.f_locals.get('self') is factory.provider:
            rows.append(dict(frame=frame.f_locals['frame'], returned=value is not None))
    return profile


def drive(m: Any, state: Any, real: Any, fixture: Any, binding: Any, factory: Any) -> dict[str, Any]:
    pipe, rec = real[:2]
    collector = m.loop.collector_loop()
    collector.RecognitionPipeline = type(pipe)
    cap = m.smoke.SyntheticCapture(real, fixture, collector.cv2, binding['provider'].o.np)
    receiver, consumer = state['postcommit_current_receiver'], state['postcommit_publication_consumer']
    originals = type(pipe).update, collector.cv2.resize, 'capture' in vars(binding['provider'])
    old_score, old_profile, views = pipe._score_zero_detector, sys.getprofile(), []
    with ExitStack() as stack:
        stack.callback(setattr, pipe, '_score_zero_detector', old_score)
        stack.callback(sys.setprofile, old_profile)
        sys.setprofile(tracked(factory, views, old_profile))
        sink = m.recording.install(stack, state, collector, fixture.subject.base)
        raw = m.raw.install(stack, collector, binding['provider'], rec.emit)
        for frame in FRAMES:
            cap.position = frame
            if frame == FRAMES[-1]:
                pipe._last_seen_next_1p = ARTIFICIAL_OLD_PAIR
            pipe._score_zero_detector = SimpleNamespace(detect=lambda pixels:
                SimpleNamespace(both_zero=frame == FRAMES[1]))
            collector.collect_lean(cap, pipe, frame, 1, K.STRIDE, K.FPS)
    K.require(originals == (type(pipe).update, collector.cv2.resize, 'capture' in vars(binding['provider'])), 'CPU_raw_restore')
    K.require(pipe._score_zero_detector is old_score and sys.getprofile() is old_profile, 'CPU_fixture_restore')
    K.require(raw.closed and raw.error is None and raw.pending is raw.active is None, 'CPU_raw_closed')
    K.require(sink.closed and not sink.errors and sink.updates == list(FRAMES), 'CPU_recording_closed')
    K.require(receiver.issued == receiver.released == 0 and not receiver.errors and receiver.active is None, 'CPU_no_event')
    K.require(len(consumer.rows) == len(FRAMES) and not consumer.errors
        and all(r['same_result_identity'] and not r['changed_sides'] for r in consumer.rows), 'CPU_outer_identity')
    K.require(views == [], 'CPU_unowned_prefix_provider_call')
    rows = fixture.recorded(real)
    raw_rows = [r for r in rows if r['kind'].startswith('directional_raw_input_')]
    skips = [r['frame_idx'] for r in raw_rows if r['kind'] == 'directional_raw_input_skipped']
    K.require(skips == [FRAMES[1]], 'CPU_original_inactive_branch')
    K.require(all(not getattr(pipe, '_tsumo_count_' + s) for s in ('1p','2p')), 'CPU_native_counter_changed')
    K.require(list(pipe._pending_tsumo_1p)==[ARTIFICIAL_OLD_PAIR] and not pipe._pending_tsumo_2p, 'CPU_legacy_append')
    K.require(len(factory.baseline_adoption.unwitnessed_appends)==1, 'CPU_unwitnessed_prefix_append')
    return dict(views=views, raw_rows=raw_rows, current_type=type(factory.controller).__name__,
        actual_update_count=len(FRAMES), issued=0, released=0, raw_restored=True, current_provider_reached=False,
        original_legacy_append_observed=True,artificial_old_next_pair=list(ARTIFICIAL_OLD_PAIR),
        artificial_pixels_scorezero=True, actual_collector_loop_prefix=True, actual_collector_append=False)


def execute(output: Path) -> Any:
    with ExitStack() as stack:
        parent = K.load('_combined_current_runner', K.CURRENT / 'run_cpu.py', stack)
        old = {name: parent.load(name, stack) for name in parent.FIXED}
        a, full = old['assembly_history'], old['run_cpu']
        m = loaded(stack)
        facade = SimpleNamespace(**(vars(a) | {'session': lambda c, modules: A.session(stack, a, c, modules)}))
        def runner(parent: Any, active: Any, binding: Any, factory: Any) -> Any:
            def run(c: Any, state: Any, real: Any, fixture: Any, inner: Any) -> Any:
                return drive(m, state, real, fixture, binding, factory)
            return run
        result = a.clone(full.execute, A=facade, FRAMES=FRAMES, runner=runner)(output)
        for name in ('HISTORY_CURRENT_PUBLICATION_STATUS.json', 'POSTCOMMIT_CONSUMER_STATUS.json'):
            value = K.read(output / name)
            K.require(value['closed'] is True and not value['errors'], 'CPU_saved_publication_closed')
        K.require(result['builder_restored'] is True, 'CPU_builder_restore')
        adopted = K.read(output/'BASELINE_ADOPTION_STATUS.json')
        K.require(adopted['closed'] and adopted['error'] is None and not adopted['adoptions'], 'CPU_adoption_closed')
        result['baseline_adoption'] = adopted
        rows = [__import__('json').loads(line) for line in (output/m.recording.SIDECAR).read_text().splitlines()]
        def no_placement(*args: Any) -> Any:
            raise AssertionError('CPU_black_fixture_must_not_place')
        history = a.clone(G.history, K=m.cpu)(rows, no_placement)
        published = a.clone(G.publication, K=m.cpu)(output, len(history['current_proofs']))
        result.update(cpu_only_recording_bounds=m.cpu.bounds(), measured_events=history | published)
        return result


if __name__ == '__main__':
    import cv2
    import torch
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    out = K.ROOT / sys.argv[1]
    K.require(out.is_dir() and out.parent.parent == K.ROOT, 'CPU_smoke_output')
    K.write(out / 'FULL_EVIDENCE.json', execute(out))
