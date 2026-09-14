"""原collectorの初期化・全ループを保持し、CPU入力供給だけをsendで区切る。"""
from __future__ import annotations
import ast
import hashlib
import inspect
from types import ModuleType
from typing import Any

SOURCE_SHA = '672963055a9fffa66531411be0a51742ee57c35de481652a77d02158ce24bee8'
INPUTS = 'cap, pipeline, start_frame, n_frames, effective_interval_frames, fps'


def extracted(source: Any) -> Any:
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA, 'collector_source_changed'
    tree = ast.parse(source.read_bytes())
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collect_lean')
    first = next(i for i,n in enumerate(original.body) if ast.unparse(n).startswith('acc = _LeanNpzAccumulator('))
    last = next(i for i,n in enumerate(original.body) if isinstance(n, ast.For) and ast.unparse(n.target)=='local_i')
    setup, loop = original.body[first:last], original.body[last]
    calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and ast.unparse(n.func)=='_process_side_lean']
    assert len(calls) == 2 and [ast.literal_eval(n.args[2]) for n in calls] == ['1P','2P']
    function = ast.parse('def continuous(start_frame: int = 0) -> object:\n    pass').body[0]
    wait = ast.parse('while True:\n    '+INPUTS+' = yield locals()\n').body[0]
    wait.body.append(loop)
    function.body = setup + [wait]
    return ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


def build(collector: Any, source: Any, overrides: dict[str, Any] | None = None) -> Any:
    defaults = {n:p.default for n,p in inspect.signature(collector.collect_lean).parameters.items()
                if p.default is not inspect.Parameter.empty}
    selected = dict(defaults, video_id='artificial-continuous-collector', start_frame=0)
    selected.update(overrides or {})
    values = dict(vars(collector), **selected)
    exec(compile(extracted(source), str(source), 'exec'), values)
    generator = values['continuous']()
    state = next(generator)
    module = ModuleType('_cpu_original_complete_collector_loop')
    module.cv2, module.TARGET_W, module.TARGET_H = collector.cv2, collector.TARGET_W, collector.TARGET_H
    module.generator, module.runtime_state, module.configuration = generator, state, selected
    def collect_lean(cap: Any, pipeline: Any, start_frame: int, n_frames: int,
                     effective_interval_frames: int, fps: float) -> None:
        # 原クラスを使うraw計装の参照先だけを同期。採録状態は再初期化しない。
        values['RecognitionPipeline'] = module.RecognitionPipeline
        module.runtime_state = generator.send((cap,pipeline,start_frame,n_frames,effective_interval_frames,fps))
    module.collect_lean = collect_lean
    return module
