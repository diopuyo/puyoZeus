"""A40のsource認証を維持したG3 video38連続履歴接続。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import inspect
import json
import os
import signal
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterator

from scripts.g3_agent_review import ROOT, VERIFY, read, save

A40 = ROOT / 'data/verify/g2_second_prefix_runtime_2026-09-14_v40'
SOURCE = ROOT / 'data/frames/video_38.mp4'
SOURCE_SHA = 'b3728078cc2e8282065e5dd78ca1c13e6a443ffdf1dc4333e3da06f0852757d3'
FIRST, END, STRIDE, FPS = 0, 36300, 2, 60
THREADS = 2
CPU_STOP = 'g3_planned_stop_after_actual_bridge'
COMMON_FILE = ROOT / 'data/verify/g2_history_publication_probe_runtime_2026-09-10_v13/common.py'


def require(value: bool, reason: str) -> None:
    """契約不一致を欠測なしの成功へ変換しない。"""
    if not value:
        raise ValueError('g3_video38:' + reason)


def patch_dict(stack: ExitStack, namespace: dict, changes: dict) -> None:
    """実関数globalsを所有scope内だけ変更し、例外でも復元する。"""
    old = {name: namespace[name] for name in changes}
    namespace.update(changes)
    stack.callback(namespace.update, old)


def bind_bounds(stack: ExitStack, main: Any, history: Any) -> list[dict]:
    """収集/記録の範囲だけを変更。scenario・資格・モデル定数は触らない。"""
    ns, seen, receipts = main.__globals__, set(), []
    modules = [ns['K'], ns['S'].K, ns['Q'].K, ns['R'].K]
    for module in modules:
        require(Path(module.__file__).resolve() == COMMON_FILE, 'common_owner')
        spaces = [vars(module)] + [fn.__globals__ for fn in vars(module).values()
            if inspect.isfunction(fn) and Path(fn.__code__.co_filename).resolve() == COMMON_FILE]
        for space in spaces:
            if id(space) in seen:
                continue
            seen.add(id(space))
            require(space['FIRST'] == 29052 and space['END'] in (36300, 36902)
                    and space['FPS'] == FPS and space['STRIDE'] == STRIDE, 'old_collection_scope')
            receipts.append(dict(kind='common', old_first=space['FIRST'], old_end=space['END']))
            patch_dict(stack, space, dict(FIRST=FIRST, END=END, FRAMES=tuple(range(FIRST, END, STRIDE))))
    space = history.HistoryRecorder.begin_frame.__globals__
    require(space['FIRST_FRAME'] == 29052 and space['END_FRAME'] == 36902, 'history_owner')
    patch_dict(stack, space, dict(FIRST_FRAME=FIRST, END_FRAME=END))
    receipts.append(dict(kind='HistoryRecorder.begin_frame', first=FIRST, end=END))
    require(ns['K'].bounds()['first_frame'] == FIRST and ns['S'].K.bounds()['end_exclusive'] == END,
            'actual_function_bounds')
    return receipts


def validate_plan(path: Path, digest: str) -> dict:
    """CLIで固定した計画と、実入力・両armの共通資材を検証する。"""
    require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'plan_sha')
    plan = read(path)
    require(plan['source_sha256'] == SOURCE_SHA and Path(plan['video_path']).resolve() == SOURCE,
            'source_scope')
    require(plan['frames'] == [FIRST, END, STRIDE] and plan['fps'] == FPS, 'frame_contract')
    require(plan['quality_gate_clear'] is False and plan['gt_scoring'] == 'NOT_SCORED_NO_CERTIFIED_GT',
            'observation_authority')
    for name, expected in (plan['entry_pins'] | plan['runtime_pins']).items():
        require(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, 'entry_changed:' + name)
    stat = SOURCE.stat()
    require(stat.st_size == plan['source_size'] and stat.st_mtime_ns == plan['source_mtime_ns'],
            'source_metadata_changed')
    return plan


def receipt_for(plan: dict, output: Path, base: Any) -> tuple[dict, dict]:
    """旧票の版情報だけを再利用し、旧run盤面を新runの真値へ渡さない。"""
    old = Path(plan['prepare_receipt'])
    require(hashlib.sha256(old.read_bytes()).hexdigest() == plan['prepare_sha256'], 'prepare_changed')
    receipt = read(old)
    require(Path(receipt['video_path']).resolve() == SOURCE, 'prepare_video_path')
    require(receipt['input_and_code_sha256'][str(SOURCE)] == SOURCE_SHA, 'source_pin_missing')
    receipt['input_and_code_sha256'].update(plan['runtime_pins'])
    base.assert_unchanged(receipt['input_and_code_sha256'])
    receipt.update(video_path=str(SOURCE), target_board={'sha256': None}, run_id=str(output),
        target_reference_disabled=True, requested_interval_sec=[0, END / FPS],
        initialization='cold_at_frame_0_same_pipeline_through_frame_36298',
        output_boundary=dict(path=str(output), exclusive_entry_directory=True,
            original_prepare_returned_before_mkdir=False, shared_live_preflight=False,
            frozen_prepare_reference=str(old)),
        expected_frame_count=len(range(FIRST, END, STRIDE)), quality_gate_clear=False,
        gt_scoring=plan['gt_scoring'], g3_plan_sha256=plan['_sha256'])
    return receipt, dict(collection_tokens=receipt['original_collection_tokens'])


@contextmanager
def capture_scope(collector: Any, state: dict) -> Iterator[None]:
    """実入力の所有・絶対frameを検査し、原runner例外時もcaptureを解放する。"""
    cv, original, captures = collector.cv2, collector.cv2.VideoCapture, []
    class Capture:
        def __init__(self, path: Any, *args: Any, **kwargs: Any) -> None:
            require(Path(path).resolve() == SOURCE, 'actual_capture_source')
            self.inner = original(str(SOURCE), *args, **kwargs)
            captures.append(self)
            self.closed = False
        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)
        def get(self, key: int) -> float:
            return float(FPS) if key == cv.CAP_PROP_FPS else self.inner.get(key)
        def set(self, key: int, value: float) -> bool:
            require(key != cv.CAP_PROP_POS_FRAMES or value == FIRST, 'seek_forbidden')
            return self.inner.set(key, value)
        def read(self) -> tuple[bool, Any]:
            before = self.inner.get(cv.CAP_PROP_POS_FRAMES)
            require(before == state['decoded_frames'] and before < END, 'decode_continuity')
            ok, image = self.inner.read()
            require(ok and image is not None, 'early_eof:' + str(before))
            require(self.inner.get(cv.CAP_PROP_POS_FRAMES) == before + 1, 'decode_position')
            state['decoded_frames'] += 1
            return ok, image
        def release(self) -> None:
            if not self.closed:
                self.inner.release()
                self.closed = True
    collector.cv2.VideoCapture = Capture
    try:
        yield
    finally:
        collector.cv2.VideoCapture = original
        errors = []
        for cap in captures:
            try:
                cap.release()
            except BaseException as error:
                errors.append(repr(error))
        state.update(captures=len(captures), captures_closed=all(c.closed for c in captures),
                     capture_release_errors=errors)


def baseline(env: Any, output: Path, receipt: dict, collector: Any, kwargs: dict) -> dict:
    """元collectorの認識/採用判定を使い、同じ新runの観測を保存する。"""
    base = env['runtime'].M.base
    with (output / 'frames.jsonl').open('x', encoding='utf-8') as stream:
        rec = base.Recorder(stream, receipt['target_board'])
        try:
            with ExitStack() as stack:
                base.instrument_pipeline(stack, collector, rec)
                base.instrument_storage(stack, collector, rec)
                base.instrument_model_load(stack, rec, receipt['input_and_code_sha256'])
                base.instrument_video(stack, collector, rec)
                try:
                    collector.collect_lean(SOURCE, output / 'not_written.npz', **kwargs)
                except base.CollectionFinished:
                    pass
                else:
                    raise RuntimeError('original_storage_boundary_not_reached')
            require(rec.frames == len(range(FIRST, END, STRIDE)) and rec.frame == END - STRIDE,
                    'baseline_coverage')
            return dict(frame_count=rec.frames, last_frame=rec.frame, pipeline=rec.pipeline_receipt)
        finally:
            stream.flush()
            os.fsync(stream.fileno())
            save(output / 'BASELINE_CLOSURE.json', dict(frame_count=rec.frames, last_frame=rec.frame,
                 model_loads=rec.model_loads, pipeline=rec.pipeline_receipt, quality_gate_clear=False))


def install_cpu_stop(stack: ExitStack, adapter: Any, state: dict) -> None:
    """既存v18と同じ原Bridge生成直後停止を、新entryの経路へ接続する。"""
    whole = adapter.A.A.A.V4.A
    original = whole.W.Bridge
    def bridge(*args: Any, **kwargs: Any) -> Any:
        state['_bridge'] = original(*args, **kwargs)
        raise RuntimeError(CPU_STOP)
    whole.W.Bridge = bridge
    stack.callback(setattr, whole.W, 'Bridge', original)


def protect_runtime(stack: ExitStack, adapter: Any, main: Any, plan: dict) -> None:
    """原targetと同じA40依存guardを実configuredへ接続する。"""
    pins = plan['runtime_pins']
    require(all(str(path) in pins for path in adapter.sources()), 'runtime_pin_coverage')
    ns = main.__globals__
    for module in {id(item): item for item in (ns['K'], ns['S'].K)}.values():
        adapter.protect(stack, module, pins)


def execute(plan: dict, output: Path, arm: str, cpu_prepare: bool, state: dict) -> dict:
    """凍結構成の実入口から原collectorへ到達する。sys.path等は全て復元する。"""
    with ExitStack() as stack:
        paths, cwd = list(sys.path), Path.cwd()
        stack.callback(sys.path.__setitem__, slice(None), paths)
        stack.callback(os.chdir, cwd)
        sys.path.insert(0, str(A40))
        import owned_adapter as adapter
        main = adapter.configured(stack)
        protect_runtime(stack, adapter, main, plan)
        if cpu_prepare:
            require(arm == 'candidate', 'cpu_prepare_candidate_only')
            install_cpu_stop(stack, adapter, state)
        ns, session = main.__globals__, main.__globals__['S']
        with session.configured() as env, session.live_scopes(env):
            base, history = env['runtime'].M.base, env['runtime'].M.A.entry.previous.history
            state['scope_bindings'] = bind_bounds(stack, main, history)
            receipt, config = receipt_for(plan, output, base)
            os.chdir(base.SNAPSHOT)
            collector = base.load_collector()
            kwargs = base.collection_arguments(collector, config)
            kwargs.update(start_sec=0.0, max_sec=END / FPS, precise_seek=False)
            require(kwargs['normalize_fps_30'] is True and kwargs['sample_interval_sec'] == 0,
                    'collector_stride_contract')
            receipt['actual_collector_kwargs'] = base.json_value(kwargs)
            receipt['recognition_variant'] = 'frozen_baseline' if arm == 'baseline' else 'A40_original_policy'
            save(output / 'PLAN.json', receipt)
            with capture_scope(collector, state):
                result = (ns['collect'](env, output, receipt, collector, kwargs) if arm == 'candidate'
                          else baseline(env, output, receipt, collector, kwargs))
            base.assert_unchanged(receipt['input_and_code_sha256'])
            return result


def arguments() -> argparse.Namespace:
    """実入口の固定引数だけを受け付ける。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=('baseline', 'candidate'), required=True)
    parser.add_argument('--cpu-prepare', action='store_true')
    return parser.parse_args()


def initialize(cpu_prepare: bool, state: dict) -> None:
    """実deviceを確認し、所有監視器のTERMでもfinally保存へ戻す。"""
    import cv2
    import torch
    def stopped(signum: int, frame: Any) -> None:
        raise RuntimeError('owned_resource_stop_SIGTERM')
    signal.signal(signal.SIGTERM, stopped)
    cv2.setNumThreads(THREADS)
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(THREADS)
    state['cuda_initialized_before'] = torch.cuda.is_initialized()
    if not cpu_prepare:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'explicit_gpu_zero')
        state['actual_gpu'] = torch.cuda.get_device_name(0)
        require('RTX 4060' in state['actual_gpu'], 'actual_gpu_not_RTX4060')


def main() -> int:
    """成功・CPU計画停止・実失敗を区別して保存し、実exitを返す。"""
    args = arguments()
    plan = validate_plan(args.plan, args.plan_sha)
    plan['_sha256'] = args.plan_sha
    output = args.output.resolve()
    require(output.is_relative_to(VERIFY.resolve()), 'output_not_D_verify')
    output.mkdir(parents=True, exist_ok=False)
    state = dict(arm=args.arm, decoded_frames=0, status='FAILED', quality_gate_clear=False,
                 started_epoch=time.time(), pid=os.getpid(), plan_sha256=args.plan_sha)
    code = 1
    try:
        initialize(args.cpu_prepare, state)
        state['summary'] = execute(plan, output, args.arm, args.cpu_prepare, state)
        require(state['decoded_frames'] == END and state['captures'] == 1
                and state['captures_closed'] and not state['capture_release_errors'], 'capture_closure')
        validate_plan(args.plan, args.plan_sha)
        state['status'], code = 'OBSERVATION_ENDED_NOT_G3_PASS', 0
    except BaseException as error:
        state['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        if args.cpu_prepare and str(error) == CPU_STOP and '_bridge' in state:
            import torch
            scope = state['_bridge'].state['whole_dependency_scope']
            state['cpu_bridge_closed'] = scope.closed and scope.error is None
            state['cuda_initialized_after'] = torch.cuda.is_initialized()
            if state['cpu_bridge_closed'] and not state['cuda_initialized_after']:
                state['status'], code = 'CPU_PLANNED_STOP', 0
    finally:
        state.pop('_bridge', None)
        if state.get('capture_release_errors'):
            state['status'], code = 'FAILED', 1
        state.update(exit_code=code, ended_epoch=time.time())
        save(output / 'ENTRY_RESULT.json', state)
    print(json.dumps(dict(status=state['status'], exit_code=code, output=str(output))))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
