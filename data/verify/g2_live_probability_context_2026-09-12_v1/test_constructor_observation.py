"""取得器の制御のみ。人工collectorによる成功を実constructor資格とは呼ばない。"""
from __future__ import annotations
from contextlib import ExitStack
import inspect
import importlib.util
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace as N
from typing import Any
import pytest
import constructor_observation as C


def sample_collect(video_path: Path, output: Path, *, flag: bool = True,
                   score_region_calibration_path: Any = None) -> Any:
    score_region_offsets = None if score_region_calibration_path is None else dict(video=video_path.name)
    pipeline = RecognitionPipeline.load_default(flag=flag, fixed=3, score_region_offsets=score_region_offsets)
    return pipeline


@pytest.mark.parametrize('calibration', [None, 'path'])
def test_kwargs_source_defaults_and_calibration(calibration: Any) -> None:
    value = C.expected_kwargs(N(collect_lean=sample_collect), Path('video_38.mp4'), Path('.'),
                              dict(score_region_calibration_path=calibration))
    assert value == dict(flag=True, fixed=3,
                        score_region_offsets=None if calibration is None else dict(video='video_38.mp4'))


def collect(env: Any, output: Any, receipt: Any, collector: Any, kwargs: Any) -> Any:
    with ExitStack() as stack:
        constructor_guard(stack, collector, env['state'], env['base'], env)
        collector.cv2.VideoCapture('existing')
        return collector.RecognitionPipeline.load_default(**kwargs)


@pytest.mark.parametrize('failure', [False, True])
def test_intercept_restore_and_preserve_original_error(failure: bool) -> None:
    calls: list[Any] = []
    original_error = ValueError('original_model_failure')
    class Pipeline:
        @classmethod
        def load_default(cls, **kwargs: Any) -> Any:
            calls.append(kwargs)
            if failure:
                raise original_error
            return cls()
    cap = N(open=True)
    cap.release = lambda: setattr(cap, 'open', False)
    cap.isOpened = lambda: cap.open
    collector = N(RecognitionPipeline=Pipeline, cv2=N(VideoCapture=lambda path: cap))
    before = inspect.getattr_static(Pipeline, 'load_default'), collector.cv2.VideoCapture
    def patch(stack: Any, obj: Any, name: str, value: Any) -> None:
        old = inspect.getattr_static(obj, name)
        stack.callback(setattr, obj, name, old)
        setattr(obj, name, value)
    namespace = dict(collect.__globals__, constructor_guard=lambda *args: None)
    original = FunctionType(collect.__code__, namespace)
    kept: dict[str, Any] = {}
    env = dict(state={}, base=N(patch=patch), factory=object())
    try:
        with pytest.raises(ValueError if failure else C.ConstructorObserved) as info:
            C.intercepted(original, kept)(env, None, None, collector, dict(flag=True))
        if failure:
            assert info.value is original_error and 'pipe' not in kept
        else:
            assert type(kept['pipe']) is Pipeline
    finally:
        C.release(kept)
    assert calls == [dict(flag=True)] and kept['captures_released']
    assert before == (inspect.getattr_static(Pipeline, 'load_default'), collector.cv2.VideoCapture)


def test_actual_kwargs_always_selects_nondecoding_seek() -> None:
    path = Path(__file__).resolve().parent.parent / 'g2_history_publication_probe_runtime_2026-09-10_v13/common.py'
    spec = importlib.util.spec_from_file_location('_constructor_actual_kwargs', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = module.actual_kwargs(dict(enable_ojama_write_accounting_guard=True,
                                      normalize_fps_30=True, sample_interval_sec=0))
    assert value['precise_seek'] is False and value['start_sec'] == module.FIRST / module.FPS


@pytest.mark.parametrize('defect', [None, 'decoded', 'identity', 'seek'])
def test_original_recorder_capture_join(tmp_path: Path, defect: str | None) -> None:
    rec = N(decoded_frame=None, frames=0)
    class Capture:
        def read(self) -> Any:
            return rec
    frame = 29052
    kept = dict(captures=[Capture()], captures_released=True,
                state=dict(atomic_journal_observer=N(history=rec)))
    if defect == 'decoded': rec.decoded_frame = frame
    elif defect == 'identity': kept['state']['atomic_journal_observer'].history = N()
    (tmp_path / 'frames.jsonl').write_text(json.dumps(dict(kind='seek', requested_frame=frame,
        actual_frame=frame + (2 if defect == 'seek' else 0))) + '\n')
    if defect is None:
        C.inspect_capture(kept, tmp_path, frame)
    else:
        with pytest.raises(AssertionError):
            C.inspect_capture(kept, tmp_path, frame)
