"""video39の凍結基準armと観測専用reset証拠。A40修復は適用しない。"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import inspect
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace as N
from typing import Any, Iterator

from scripts import g3_baseline_observers as O
from scripts import g3_source_models as M
from scripts import g3_video38_entry as G
from scripts.g3_model_process import loaded

ROOT = G.ROOT
BASE = ROOT / 'scripts/diagnose_video38_confirmed_collapse_v1.py'
BASE_SHA = 'b20525d5b7423a34125a45b11f76aea68d0942276109463007dfd991e46fa144'
SOURCE_NAME, FPS, STRIDE, END = 'video_39', 30, 1, 11878
REPAIR = G.VERIFY / 'g3_repair_2026-09-15_v1'
REFERENCE_PLAN = REPAIR / 'VIDEO38_ADMISSION_PLAN.json'
REFERENCE_SHA = '677eb56c099884075a2567c525a69f3116c71c08335ca0974bfca9f14cd1b0f4'
CALIBRATION = ROOT / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/data/verify/score_region_calibration_48pilot_2026-08-29/video_39_score_region_v1.json'
CALIBRATION_SHA = '14aa867d17a5b874abf6411d4420dcbe643a4cb4d262e82b9c0c2fafca6a5a18'
CPU_STOP = 'g3_baseline_planned_stop_before_cnn_constructor'
CPU_SECONDS, GPU_SECONDS = 300, 2700
CLOCK_EVIDENCE = REPAIR / 'VIDEO39_CLOCK_EVIDENCE_V2.json'
CLOCK_SHA = '16ef64fc7c6b53f1596de0e77eb4cfa4f66556b0925b8dc2098e5a0bc74e03ff'


def source_contract() -> dict:
    """既定video39契約だけを受け入れ、非整数fpsへの一般化をしない。"""
    prereg = M.document(M.PREREG, M.PREREG_SHA)
    item = M.one(prereg['sources'], 'source_video_id', SOURCE_NAME)
    source, scope = item['source'], item['required_continuous_processing']
    G.require((source['time_base_numerator'], source['time_base_denominator']) == (1, FPS), 'baseline_clock')
    G.require(scope['start_frame'] == 0 and scope['end_frame_exclusive'] == END
              and scope['seek_between_windows'] is False, 'baseline_scope')
    G.require(int(END / FPS * FPS) == END, 'baseline_end_rounding')
    return source


def prepare(path: Path) -> str:
    """旧prepareの設定/資材だけを引用し、新runのreceiptは別に作る。"""
    source = source_contract()
    prior = M.document(REFERENCE_PLAN, REFERENCE_SHA)
    receipt = M.document(Path(prior['prepare_receipt']), prior['prepare_sha256'])
    G.require(receipt['video_path'] == prior['video_path'], 'baseline_reference_source')
    calibration = M.document(CALIBRATION, CALIBRATION_SHA)
    G.require(calibration['source_video_id'] == SOURCE_NAME
              and calibration['source_video_sha256'] == source['source_video_sha256'], 'baseline_calibration')
    tokens = list(receipt['original_collection_tokens'])
    G.require(tokens.count('--score-region-calibration') == 1, 'baseline_calibration_option')
    tokens[tokens.index('--score-region-calibration') + 1] = str(CALIBRATION)
    video = ROOT / source['source_video_path']
    pins = dict(receipt['input_and_code_sha256'])
    G.require(pins[str(BASE)] == BASE_SHA, 'baseline_reference_base')
    pins.update({str(video): source['source_video_sha256'], str(CALIBRATION): CALIBRATION_SHA})
    modules = (Path(__file__), Path(O.__file__), Path(G.__file__), Path(M.__file__),
               ROOT / 'scripts/g3_model_process.py', ROOT / 'scripts/g3_admission.py', O.EPOCH, M.PREREG, CLOCK_EVIDENCE)
    entry_pins = {str(p.resolve()): M.digest(p) for p in modules}
    stat = video.stat()
    plan = dict(video_path=str(video), source_sha256=source['source_video_sha256'],
                source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                frames=[0, END, STRIDE], fps=FPS, source=source,
                source_id='sha256:' + source['source_video_sha256'],
                entry_pins=entry_pins, runtime_pins=pins, tokens=tokens,
                reference_prepare=prior['prepare_receipt'], reference_sha=prior['prepare_sha256'],
                gt_scoring='NOT_SCORED_NO_CERTIFIED_GT', quality_gate_clear=False,
                mode='frozen_baseline_observation_only_no_A40', reset_repair_forbidden=True,
                latency_target_ms=prior['latency_target_ms'], latency_ceiling_ms=prior['latency_ceiling_ms'])
    G.require(path.resolve().is_relative_to(G.VERIFY.resolve()), 'baseline_plan_output')
    G.save(path, plan)
    return M.digest(path)


def bind(stack: ExitStack) -> dict:
    """別process内の旧capture/baseline参照へ認証済み契約を束縛する。"""
    source = source_contract()
    G.patch_dict(stack, vars(G), dict(SOURCE=ROOT / source['source_video_path'],
                 SOURCE_SHA=source['source_video_sha256'], FIRST=0, END=END, STRIDE=STRIDE, FPS=FPS))
    return source


def arguments(collector: Any, tokens: list[str], video: Path) -> tuple[dict, dict]:
    """元CLI parserで実videoも捕捉し、指定sourceを捨てない。"""
    original, argv, caught = collector.collect_lean, sys.argv, {}
    def capture(selected: Path, output: Path, **kwargs: Any) -> int:
        caught.update(video=selected, kwargs=kwargs)
        return 0
    try:
        collector.collect_lean = capture
        sys.argv = ['frozen-collector', '--video', str(video), '--out-npz', 'diagnostic-unused.npz', *tokens]
        collector.main()
    finally:
        collector.collect_lean, sys.argv = original, argv
    G.require(Path(caught['video']).resolve() == video.resolve(), 'baseline_actual_cli_video')
    kwargs = caught['kwargs']
    kwargs.update(start_sec=0.0, max_sec=END / FPS, precise_seek=False, sample_interval_frames=STRIDE)
    G.require(kwargs['normalize_fps_30'] is True and kwargs['sample_interval_sec'] == 0, 'baseline_sampling')
    return kwargs, dict(video=str(caught['video']), actual_kwargs=kwargs)


def native_clock(cv: Any, source: dict, state: dict) -> float:
    """同じsourceとOpenCV buildで実測した平均値を固定し、評価時計と分ける。"""
    evidence = M.document(CLOCK_EVIDENCE, CLOCK_SHA)
    G.require(evidence['source_verified_sha256'] == source['source_video_sha256'], 'native_clock_source')
    build_sha = hashlib.sha256(cv.getBuildInformation().encode()).hexdigest()
    G.require(cv.__version__ == evidence['opencv']['version']
              and build_sha == evidence['opencv']['build_sha256'], 'native_clock_build')
    fps = evidence['observed_opencv_fps']
    state['clock_evidence_sha256'] = CLOCK_SHA
    state['native_fps_delta'] = fps - FPS
    state['pts_maximum_delta_seconds'] = evidence['maximum_absolute_delta_seconds']
    return fps


@contextmanager
def checked_capture(collector: Any, source: dict, state: dict) -> Iterator[None]:
    """FPS上書きより前の実コンテナ値を検査し、失敗時も実captureを閉じる。"""
    cv, original = collector.cv2, collector.cv2.VideoCapture
    expected_fps = native_clock(cv, source, state)
    def create(*args: Any, **kwargs: Any) -> Any:
        cap = original(*args, **kwargs)
        try:
            actual = dict(fps=cap.get(cv.CAP_PROP_FPS), frames=cap.get(cv.CAP_PROP_FRAME_COUNT),
                          width=cap.get(cv.CAP_PROP_FRAME_WIDTH), height=cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            state['actual_container'] = actual
            G.require(cap.isOpened() and actual == dict(fps=expected_fps, frames=source['frame_count'],
                       width=source['width'], height=source['height']), 'baseline_actual_container')
            return cap
        except BaseException:
            try:
                cap.release()
            except BaseException as error:
                state['preflight_release_error'] = repr(error)
            raise
    collector.cv2.VideoCapture = create
    try:
        with G.capture_scope(collector, state):
            yield
    finally:
        collector.cv2.VideoCapture = original


def instrumentation(base: Any, stack: ExitStack, state: dict, source_id: str,
                    cpu_prepare: bool) -> None:
    """元基準の計装後に観測だけを設置。CPUは実constructor直前で停止する。"""
    original = base.instrument_pipeline
    def install(inner: Any, collector: Any, rec: Any) -> None:
        original(inner, collector, rec)
        O.install(inner, collector, rec, state, base.patch, source_id=source_id)
        if cpu_prepare:
            def stop(cls: Any, *args: Any, **kwargs: Any) -> None:
                state['actual_constructor_kwargs'] = base.json_value(kwargs)
                caller = sys._getframe(1)
                G.require(caller.f_code is collector.collect_lean.__code__
                          and caller.f_globals is vars(collector), 'baseline_constructor_caller')
                values = caller.f_locals
                keys = ('video_path', 'fps', 'start_frame', 'end_frame', 'effective_interval_frames')
                state['actual_loop'] = base.json_value({key: values[key] for key in keys})
                G.require(values['video_path'] == G.SOURCE and values['fps'] == FPS
                          and values['start_frame'] == 0 and values['end_frame'] == END
                          and values['effective_interval_frames'] == STRIDE, 'baseline_actual_loop')
                raise RuntimeError(CPU_STOP)
            base.patch(inner, collector.RecognitionPipeline, 'load_default', classmethod(stop))
    base.instrument_pipeline = install
    stack.callback(setattr, base, 'instrument_pipeline', original)


def execute(plan: dict, output: Path, state: dict, cpu_prepare: bool) -> dict:
    """元baselineだけを通し、A40 configured/Driver/候補修復は呼ばない。"""
    with ExitStack() as stack:
        source = bind(stack)
        paths, cwd = list(sys.path), Path.cwd()
        stack.callback(sys.path.__setitem__, slice(None), paths)
        stack.callback(os.chdir, cwd)
        base = stack.enter_context(loaded(BASE, BASE_SHA))
        collector = base.load_collector()
        def release_collector() -> None:
            if sys.modules.get(collector.__name__) is collector:
                del sys.modules[collector.__name__]
        stack.callback(release_collector)
        os.chdir(base.SNAPSHOT)
        kwargs, cli = arguments(collector, plan['tokens'], G.SOURCE)
        state['actual_cli'] = base.json_value(cli)
        receipt = dict(video_path=str(G.SOURCE), source_id=plan['source_id'], run_id=str(output),
                       target_board={'sha256': None}, target_reference_disabled=True,
                       input_and_code_sha256=plan['runtime_pins'] | plan['entry_pins'],
                       actual_collector_kwargs=base.json_value(kwargs),
                       mode=plan['mode'], quality_gate_clear=False)
        G.save(output / 'PLAN.json', receipt)
        instrumentation(base, stack, state, plan['source_id'], cpu_prepare)
        refs = tuple(inspect.getattr_static(collector.RecognitionPipeline, name)
                     for name in ('update', 'load_default', 'reset', '_update_score_tracker'))
        try:
            with checked_capture(collector, source, state):
                return G.baseline({'runtime': N(M=N(base=base))}, output, receipt, collector, kwargs)
        finally:
            state['references_restored'] = refs == tuple(inspect.getattr_static(collector.RecognitionPipeline, name)
                         for name in ('update', 'load_default', 'reset', '_update_score_tracker'))


def main() -> int:
    """計画CPU停止と実観測終了を分離し、元guardと失敗票を保持する。"""
    args = G.arguments()
    G.require(args.arm == 'baseline', 'video39_baseline_only')
    output = args.output.resolve()
    G.require(output.is_relative_to(G.VERIFY.resolve()), 'baseline_output')
    output.mkdir(parents=True, exist_ok=False)
    state = dict(output=output, decoded_frames=0, status='FAILED', quality_gate_clear=False,
                 pid=os.getpid(), started_epoch=time.time())
    code, plan = 1, None
    def timed_out(signum: int, frame: Any) -> None:
        raise RuntimeError('g3_baseline_wall_timeout')
    try:
        signal.signal(signal.SIGALRM, timed_out)
        signal.alarm(CPU_SECONDS if args.cpu_prepare else GPU_SECONDS)
        with ExitStack() as stack:
            bind(stack)
            plan = G.validate_plan(args.plan, args.plan_sha)
        G.initialize(args.cpu_prepare, state)
        state['summary'] = execute(plan, output, state, args.cpu_prepare)
        G.require(state['decoded_frames'] == END and state['captures_closed']
                  and state['references_restored'], 'baseline_coverage_or_cleanup')
        state['status'], code = 'BASELINE_OBSERVATION_ENDED_NOT_G3_PASS', 0
    except BaseException as error:
        state['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        if args.cpu_prepare and str(error) == CPU_STOP:
            import torch
            state['cuda_initialized_after'] = torch.cuda.is_initialized()
            if state.get('captures_closed') and state.get('references_restored') and not state['cuda_initialized_after']:
                state['status'], code = 'CPU_PLANNED_STOP', 0
    finally:
        if plan is not None:
            try:
                with ExitStack() as stack:
                    bind(stack)
                    G.validate_plan(args.plan, args.plan_sha)
            except BaseException as error:
                state['end_guard_error'], state['status'], code = repr(error), 'FAILED', 1
        signal.alarm(0)
        state.pop('output', None)
        state.pop(O.A.KEY, None)
        state.update(exit_code=code, ended_epoch=time.time())
        G.save(output / 'ENTRY_RESULT.json', state)
    print(json.dumps(dict(status=state['status'], exit_code=code, output=str(output))))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
