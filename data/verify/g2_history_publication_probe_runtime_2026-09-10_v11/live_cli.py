"""履歴current公開prefixのprepare/live/finalize。実GPU開始はlauncherを実行する親だけ。"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any
import common as K
import closure as Q
import recording as R
import session as S


def refs(collector: Any) -> tuple[Any, ...]:
    cls, acc = collector.RecognitionPipeline, collector._LeanNpzAccumulator
    return (cls.update, cls._step_side, inspect.getattr_static(cls, 'load_default'),
            acc.append, acc.save, collector.cv2.VideoCapture)


def constructor_guard(stack: Any, collector: Any, state: Any, base: Any) -> None:
    cls = collector.RecognitionPipeline
    original = inspect.getattr_static(cls, 'load_default')
    def load(inner: Any, *args: Any, **kwargs: Any) -> Any:
        K.require('actual_constructor' not in state, 'duplicate_pipeline_constructor')
        value = original.__func__(inner, *args, **kwargs)
        K.require(value._enable_ojama_write_accounting_guard is True, 'actual_raw_capture_flag')
        state['actual_constructor'] = dict(calls=1, raw_capture_flag=True,
            input_kwargs=base.json_value(kwargs), original_return_preserved=True)
        return value
    base.patch(stack, cls, 'load_default', classmethod(load))


def collect(env: Any, output: Path, receipt: Any, collector: Any, kwargs: Any) -> dict[str, Any]:
    runtime, base = env['runtime'], env['runtime'].M.base
    history_module = runtime.M.A.entry.previous.history
    state: dict[str, Any] = dict(output=output, sink=None)
    original, stream = refs(collector), (output / 'frames.jsonl').open('x', encoding='utf-8')
    rec, started = history_module.HistoryRecorder(stream, receipt['target_board']), time.perf_counter()
    try:
        with env['a'].directional(env['x'], runtime, env['addon'], env['factory'], K.SOURCE,
                                 output.resolve().as_posix()) as binding, ExitStack() as stack:
            runtime.M.instrument(stack, collector, rec, receipt, state)
            base.instrument_storage(stack, collector, rec)
            base.instrument_model_load(stack, rec, receipt['input_and_code_sha256'])
            base.instrument_video(stack, collector, rec)
            constructor_guard(stack, collector, state, base)
            sink = R.install(stack, state, collector, base)
            raw = K.load('_history_raw_bridge', K.VERIFY / 'g2_directional_raw_input_2026-09-09_v3/bridge.py', stack)
            state['raw_input_bridge'] = raw.install(stack, collector, binding['provider'], rec.emit)
            handles = R.stream_handles(state) | {'frames.jsonl': stream, R.SIDECAR: sink.stream}
            try:
                collector.collect_lean(Path(receipt['video_path']), output / 'not_written.npz', **kwargs)
            except base.CollectionFinished:
                pass
            else:
                raise RuntimeError('original_storage_boundary_not_reached')
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        K.require(refs(collector) == original, 'original_pipeline_storage_video_not_restored')
        summary = Q.finish(env, state, rec, sink, handles, binding)
        return summary | dict(collection_elapsed_sec=time.perf_counter() - started, references_restored=True,
            actual_constructor=state['actual_constructor'])
    finally:
        stream.close()
        K.write(output / 'CLOSURE_STATE.json', dict(frame_count=rec.frames, last_frame=rec.frame,
            original_refs_restored=refs(collector) == original, state_keys=sorted(state),
            errors={key: repr(getattr(value, 'sticky_error', None)) for key, value in state.items()
                    if getattr(value, 'sticky_error', None) is not None}))


def run_live(output: Path, owned: list[Path]) -> dict[str, Any]:
    K.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'explicit_single_GPU_required')
    with S.configured() as env, S.live_scopes(env):
        receipt, config = S.prepare_output(env, output, owned)
        base = env['runtime'].M.base
        initial_cwd, paths = Path.cwd(), list(sys.path)
        try:
            os.chdir(base.SNAPSHOT)
            collector = base.load_collector()
            original = base.collection_arguments(collector, config)
            kwargs = K.actual_kwargs(original)
            receipt.update(original_collector_kwargs=base.json_value(original),
                actual_collector_kwargs=base.json_value(kwargs), diagnostic_argv=sys.argv,
                initial_src_modules=env['runtime'].M.A.entry.previous.history.frozen_modules())
            K.write(output / 'PLAN.json', receipt)
            summary = collect(env, output, receipt, collector, kwargs)
        finally:
            os.chdir(initial_cwd)
            sys.path[:] = paths
        base.assert_unchanged(receipt['input_and_code_sha256'])
        summary['guards_unchanged'] = True
    summary.update(outer_configuration_restored=True, original_three_scopes_held_through_collection=True)
    K.write(output / 'SUMMARY.json', summary)
    return dict(status='engine_closed_await_parent_finalize',
        current_event_observed=summary['goal']['current_event_observed'],
        outer_publication_observed=summary['goal']['outer_publication_observed'])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'live', 'finalize'), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--child-exit', type=int)
    parser.add_argument('--resource-exit', type=int)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.mode == 'finalize':
        result = Q.finalize(output, args.child_exit, args.resource_exit, Path(str(output) + '.resources.jsonl'))
        print(json.dumps(result), flush=True)
        return 0
    K.require(output.parent == (K.VERIFY if args.mode == 'live' else K.ROOT), 'new_output_parent')
    if args.mode == 'live':
        K.require(output.name.startswith('video38_history_publication_probe_'), 'live_output_namespace')
    K.require(not output.exists() and not output.is_symlink(), 'exclusive_output')
    owned: list[Path] = []
    started, code = time.perf_counter(), 0
    try:
        import cv2
        import torch
        cv2.setNumThreads(2)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        result = S.preflight(output, owned) if args.mode == 'prepare' else run_live(output, owned)
    except BaseException:
        result, code = dict(error=traceback.format_exc()), 1
        if output not in owned:
            output.mkdir(exist_ok=False)
            owned.append(output)
    result.update(exit_code=code, seconds=time.perf_counter() - started, pid=os.getpid())
    K.write(output / 'ENTRY_RESULT.json', result)
    if args.mode == 'live' and code == 0:
        Q.seal(output, K.read(output / 'SUMMARY.json'), K.read(output / 'PLAN.json'))
    if args.mode == 'prepare' and code == 0:
        K.write(output / 'PREPARE_COMPLETE.json', dict(sha256={p.name: K.sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
