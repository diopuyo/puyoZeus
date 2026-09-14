"""原collectの実生成完了直後だけ停止する。更新/全終了を成功扱いしない。"""
from __future__ import annotations
import ast
import inspect
import json
from pathlib import Path
from types import FunctionType
from typing import Any


class ConstructorObserved(RuntimeError):
    """実constructor返却と設置完了を取得した後の計画停止。"""


def expected_kwargs(collector: Any, video: Path, output: Path, kwargs: Any) -> dict[str, Any]:
    function = collector.collect_lean
    bound = inspect.signature(function).bind(video, output / 'not_written.npz', **kwargs)
    bound.apply_defaults()
    namespace = dict(function.__globals__, **bound.arguments)
    tree = ast.parse(inspect.getsource(function))
    nodes = [node for node in tree.body[0].body if isinstance(node, ast.Assign)]
    offsets = [node for node in nodes if any(isinstance(t, ast.Name) and t.id == 'score_region_offsets'
                                            for t in node.targets)]
    pipeline = [node for node in nodes if any(isinstance(t, ast.Name) and t.id == 'pipeline'
                                             for t in node.targets)]
    assert len(offsets) == len(pipeline) == 1, 'constructor_source_assignments'
    call = pipeline[0].value
    assert isinstance(call, ast.Call) and ast.unparse(call.func) == 'RecognitionPipeline.load_default'
    assert not call.args and all(item.arg is not None for item in call.keywords)
    def value(node: ast.AST) -> Any:
        return eval(compile(ast.Expression(body=node), inspect.getsourcefile(function), 'eval'), namespace)
    namespace['score_region_offsets'] = value(offsets[0].value)
    return {item.arg: value(item.value) for item in call.keywords}


def intercepted(original: Any, kept: dict[str, Any]) -> Any:
    original_guard = original.__globals__['constructor_guard']
    def guard(stack: Any, collector: Any, state: Any, base: Any, env: Any = None) -> None:
        original_guard(stack, collector, state, base, env)
        kept.update(state=state, factory=env['factory'], collector=collector)
        cls = collector.RecognitionPipeline
        prior = inspect.getattr_static(cls, 'load_default')
        capture = collector.cv2.VideoCapture
        def video(*args: Any, **kwargs: Any) -> Any:
            cap = capture(*args, **kwargs)
            kept.setdefault('captures', []).append(cap)
            return cap
        def load(inner: Any, *args: Any, **kwargs: Any) -> Any:
            assert 'pipe' not in kept, 'constructor_observation_duplicate'
            result = prior.__func__(inner, *args, **kwargs)
            kept.update(pipe=result, supplied_kwargs=kwargs)
            raise ConstructorObserved('actual_constructor_return_observed')
        base.patch(stack, collector.cv2, 'VideoCapture', video)
        base.patch(stack, cls, 'load_default', classmethod(load))
    return FunctionType(original.__code__, dict(original.__globals__, constructor_guard=guard),
                         original.__name__, original.__defaults__, original.__closure__)


def release(kept: dict[str, Any]) -> None:
    for cap in kept.get('captures', []):
        cap.release()
    kept['captures_released'] = all(not cap.isOpened() for cap in kept.get('captures', []))


def inspect_capture(kept: dict[str, Any], output: Path, first_frame: int) -> None:
    assert len(kept['captures']) == 1 and kept['captures_released']
    cap = kept['captures'][0]
    recorder = inspect.getclosurevars(type(cap).read).nonlocals['rec']
    assert recorder is kept['state']['atomic_journal_observer'].history, 'constructor_original_recorder'
    assert recorder.decoded_frame is None and recorder.frames == 0, 'constructor_decoded_or_updated'
    rows = [json.loads(line) for line in (output / 'frames.jsonl').read_text().splitlines()]
    seeks = [row for row in rows if row.get('kind') == 'seek']
    assert len(seeks) == 1 and seeks[0]['requested_frame'] == seeks[0]['actual_frame'] == first_frame


def gpu_receipt() -> dict[str, Any]:
    """原scopeの解放がvenv内moduleを外す前に、実CUDA状態を値として保持する。"""
    import sys
    torch = sys.modules.get('torch')
    assert torch is not None and torch.cuda.is_initialized(), 'original_cuda_initialization_missing'
    return dict(gpu=True, gpu_device=torch.cuda.get_device_name(0), video_frame_inference=False)


def inspect_closed(main: Any, kept: dict[str, Any], expected: Any, output: Path) -> dict[str, Any]:
    import sys
    state, factory = kept['state'], kept['factory']
    common = main.__globals__['K']
    receipt = state['actual_constructor']
    assert receipt['calls'] == 1 and receipt['raw_capture_flag'] is True
    actual = state['repeated_firing_constructor']
    assert actual['attempts'] == 1 and actual['installed'] and actual['closed'] and actual['references_restored']
    assert actual['qualification']['qualified_restored'] and actual['pipe_identity'] == id(kept['pipe'])
    json_value = state['live_history_sink'].base.json_value
    assert kept['supplied_kwargs'] == expected, 'constructor_raw_kwargs'
    assert receipt['input_kwargs'] == json_value(expected) == json_value(kept['supplied_kwargs'])
    assert sys.modules['_private_live_finalizer'].context(state) is factory
    for refs in (state['rolling_prefix_references'], factory.controller.empty_runtime_references):
        assert refs['installed'] and refs['closed'] and refs['restored']
    context = state['live_empty_reset_context']
    status = sys.modules['proof_status']
    closers = [vars(cls)['close'] for cls in type(context).__mro__ if 'close' in vars(cls)]
    assert any(fn.__globals__.get('close') is status.close for fn in closers), 'actual_status_parent'
    saved = common.read(output / 'LIVE_EMPTY_RESET.json')
    closure = common.read(output / 'CLOSURE_STATE.json')
    assert saved['error'] is None and saved['baseline_recovered'] is False
    assert closure['frame_count'] == 0 and closure['original_refs_restored']
    assert closure['pipeline']['board_cnn_device'] == 'cuda:0' and closure['model_loads']
    inspect_capture(kept, output, common.FIRST)
    return dict(actual_constructor=True, kwargs_equal=True, actual_status_parent=True, references_restored=True,
                captures_released=True, updates=0, actual_video_frames=False, probability_activated=False,
                repeat_verify_reached=False, full_probability_finalizer_verified=False, quality_gate_clear=False)
