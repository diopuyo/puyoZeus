"""原whole collectの最初の4実更新で停止する。対象手/M1/全終了の合格ではない。"""
from __future__ import annotations
from contextlib import ExitStack
import json
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback
from types import FunctionType
from typing import Any
import whole_live_adapter as A
import constructor_observation as C

ROOT = Path(__file__).resolve().parent
UPDATES = 4


class LimitedObserved(RuntimeError):
    """指定実更新の保存完了後だけ発行する計画停止。"""


def intercepted(original: Any, kept: dict) -> Any:
    previous = original.__globals__['constructor_guard']
    def guard(stack: Any, collector: Any, state: Any, base: Any, env: Any = None) -> None:
        previous(stack, collector, state, base, env)
        bridge = state[A.W.KEY]
        kept.update(state=state, bridge=bridge)
        consumer = bridge.consumer
        def completed(value: Any, frame: int) -> None:
            consumer(value, frame)
            if len(value.completed_frames) != UPDATES:
                return
            receipt = dict(producer=value.capture.snapshot(), physical_count=value.physical._observed_frame_count,
                           actual_constructor=state['actual_constructor'], full_video_verified=False,
                           m1_session_created='belief_m1_session' in state, quality_gate_clear=False)
            save_receipt(state['output'] / 'WHOLE_LIMITED_INPUT.json', receipt)
            raise LimitedObserved('whole_observed_prefix_completed')
        bridge.consumer = completed
        kept['consumer'] = completed
        capture = collector.cv2.VideoCapture
        def video(*args: Any, **kwargs: Any) -> Any:
            result = capture(*args, **kwargs)
            kept.setdefault('captures', []).append(result)
            return result
        base.patch(stack, collector.cv2, 'VideoCapture', video)
    selected = FunctionType(original.__code__, dict(original.__globals__, constructor_guard=guard),
                        original.__name__, original.__defaults__, original.__closure__)
    selected.__kwdefaults__ = original.__kwdefaults__
    return selected


def save_receipt(path: Path, receipt: dict) -> None:
    """非有限値を拒否し、排他保存した原票を読み戻して照合する。"""
    raw = json.dumps(receipt, allow_nan=False, sort_keys=True).encode('utf-8')
    with path.open('xb') as stream:
        stream.write(raw)
    assert path.read_bytes() == raw, 'limited_receipt_readback'


def inspect_closed(main: Any, kept: dict, expected: dict, output: Path, base: Any) -> dict:
    common, state, bridge = main.__globals__['K'], kept['state'], kept['bridge']
    closure = json.loads((output / 'CLOSURE_STATE.json').read_bytes())
    limited = json.loads((output / 'WHOLE_LIMITED_INPUT.json').read_bytes())
    frames = tuple(common.FRAMES[:UPDATES])
    assert kept['captures_released'] and len(kept['captures']) == 1
    assert tuple(bridge.completed_frames) == frames and bridge.closed and bridge.capture.closed
    assert state['whole_dependency_scope'].closed and state['whole_dependency_scope'].error is None
    assert isinstance(bridge.error, LimitedObserved) and bridge.restore_error is None
    assert bridge.failure_save_error is None and bridge.consumer is kept['consumer']
    assert state[A.DRIVER.KEY].closed and state[A.DRIVER.KEY].session is None
    assert closure['frame_count'] == UPDATES and closure['last_frame'] == frames[-1]
    assert closure['original_refs_restored'] and closure['pipeline']['board_cnn_device'] == 'cuda:0'
    assert len(closure['model_loads']) == 2
    assert limited['producer']['observed_count'] == limited['physical_count'] == UPDATES
    assert bridge.capture.tail.count == UPDATES * len(common.SIDES)
    repeated = json.loads((output / 'REPEATED_FIRING.json').read_bytes())
    assert all(repeated[key] for key in ('installed', 'closed', 'references_restored'))
    reset = json.loads((output / 'LIVE_EMPTY_RESET.json').read_bytes())
    assert reset['error'] is None and reset['baseline_recovered'] is False
    assert limited['actual_constructor']['calls'] == 1
    assert limited['actual_constructor']['input_kwargs'] == base.json_value(expected)
    assert not limited['m1_session_created']
    return dict(status='LIMITED_REAL_PREFIX_VERIFIED', actual_frames=list(frames),
                actual_constructor=True, physical_updates=UPDATES, accounting_updates=UPDATES,
                metadata_rows=bridge.capture.tail.count, full_video_verified=False,
                actual_m1_verified=False, quality_gate_clear=False)


def collect(main: Any, env: dict, output: Path, receipt: dict, config: Any) -> dict:
    base, kept = env['runtime'].M.base, {}
    previous, paths = Path.cwd(), list(sys.path)
    try:
        os.chdir(base.SNAPSHOT)
        collector = base.load_collector()
        kwargs = main.__globals__['K'].actual_kwargs(base.collection_arguments(collector, config))
        assert kwargs['precise_seek'] is False
        expected = C.expected_kwargs(collector, Path(receipt['video_path']), output, kwargs)
        try:
            intercepted(main.__globals__['collect'], kept)(env, output, receipt, collector, kwargs)
        except LimitedObserved as error:
            assert str(error) == 'whole_observed_prefix_completed'
        else:
            raise AssertionError('limited_whole_stop_not_reached')
        finally:
            C.release(kept)
        return inspect_closed(main, kept, expected, output, base)
    finally:
        os.chdir(previous)
        sys.path[:] = paths


def execute(output: Path) -> dict:
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert not any(name == 'src' or name.startswith('src.') for name in sys.modules)
    with ExitStack() as stack:
        main = A.configured(stack)
        session = main.__globals__['S']
        with session.configured() as env, session.live_scopes(env):
            receipt, config = session.prepare_output(env, output, [])
            for path in (Path(__file__), ROOT / 'constructor_observation.py'):
                receipt['input_and_code_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            main.__globals__['K'].write(output / 'PLAN.json', receipt)
            result = collect(main, env, output, receipt, config)
            env['runtime'].M.base.assert_unchanged(receipt['input_and_code_sha256'])
            result.update(C.gpu_receipt(), video_frame_inference=True)
            main.__globals__['K'].write(output / 'WHOLE_LIMITED_CLOSED.json', result)
    return result | dict(outer_configuration_restored=True)


def main() -> int:
    output = ROOT / 'whole_observation_v1'
    assert not output.exists(), 'exclusive_whole_observation'
    paths = [ROOT / name for name in A.FILES] + [Path(__file__), ROOT / 'constructor_observation.py']
    raw = {path: path.read_bytes() for path in paths}
    started, code = time.perf_counter(), 0
    try:
        result = execute(output)
    except BaseException:
        result, code = dict(error=traceback.format_exc(), quality_gate_clear=False), 1
        output.mkdir(exist_ok=True)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir(exist_ok=False)
    for path, value in raw.items():
        with (snapshot / path.name).open('xb') as stream:
            stream.write(value)
    unchanged = all(path.read_bytes() == value for path, value in raw.items())
    result.update(exit_code=code, seconds=time.perf_counter() - started, source_unchanged=unchanged,
                  source_sha256={str(path): hashlib.sha256(value).hexdigest() for path, value in raw.items()})
    with (output / 'RESULT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)
    return code if unchanged else 1


if __name__ == '__main__':
    raise SystemExit(main())
