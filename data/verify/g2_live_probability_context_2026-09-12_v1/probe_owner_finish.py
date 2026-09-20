"""元105更新/計画停止/解除後に、原samecallと保持archiveの実終了経路を検査する。"""
from __future__ import annotations
import ast
from contextlib import ExitStack
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any
import probe_positive_v3 as V
import probability_owner as P

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1'


def original_driver() -> Any:
    """元driverの成功/失敗/解除検査を維持し、保存名だけ新runへ分離する。"""
    tree = ast.parse(inspect.getsource(V.main))
    hits = [node for node in ast.walk(tree) if isinstance(node, ast.Constant)
            and node.value == 'POSITIVE_CORE_PROBE_v3.json']
    assert len(hits) == 1
    hits[0].value = 'POSITIVE_CORE_PROBE_v68.json'
    namespace = dict(vars(V))
    exec(compile(ast.fix_missing_locations(tree), V.__file__, 'exec'), namespace)
    return namespace['main']


def closed_samecall(kept: dict[str, Any]) -> dict[str, Any]:
    state, factory, original = kept['state'], kept['factory'], kept['original']
    lease = state['repeat_scope_guard'].reset_lease
    result = P.verify(original, state, lease, factory)
    assert result['original_samecall'] == lease.empty_evidence.result
    completion = lease.empty_evidence.parts.completion
    assert type(completion) is ModuleType and 'verify' in vars(completion)
    fields = dict(vars(completion))
    previous = completion.verify
    rejected = False
    try:
        completion.verify = lambda *args: None
        try:
            P.verify(original, state, lease, factory)
        except AssertionError as error:
            assert str(error) == 'samecall_global_identity', 'unexpected_samecall_rejection'
            rejected = True
    finally:
        completion.verify = previous
    assert rejected and completion.verify is previous, 'original_samecall_code_guard'
    assert set(vars(completion)) == set(fields) and all(vars(completion)[key] is value for key, value in fields.items())
    assert P.verify(original, state, lease, factory) == result
    return dict(original_samecall=result, code_guard_rejected=True, original_callable_restored=True,
                actual_retained_archive_verified=True, factory_id=id(factory),
                intentional_original_stop=True, actual_video=False, full_main_finish_verified=False,
                model_inference_performed=False, full_probability_finalizer_verified=False, quality_gate_clear=False)


def main() -> None:
    paths = (Path(P.__file__), Path(V.__file__), Path(__file__))
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    kept: dict[str, Any] = {}
    with ExitStack() as stack:
        prior = V.positive
        def positive(boundary: Any, refs: dict[str, Any]) -> dict[str, Any]:
            result = prior(boundary, refs)
            previous = list(sys.path)
            try:
                sys.path.insert(0, str(BASE))
                original = importlib.import_module('retired_completion')
            finally:
                sys.path[:] = previous
            lease = boundary.state['repeat_scope_guard'].reset_lease
            assert type(lease) is original.E.Lease
            kept.update(state=boundary.state, factory=boundary.context['factory'], original=original)
            return result
        stack.callback(setattr, V, 'positive', prior)
        V.positive = positive
        original_driver()()
    if '--preflight' in sys.argv:
        assert not kept and before == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        return
    assert kept, 'original_positive_target_not_reached'
    assert (BASE / sys.argv[1]).resolve() == Path(kept['state']['output']).resolve()
    assert (BASE / sys.argv[1]).is_dir(), 'original_output_not_saved'
    try:
        report = closed_samecall(kept)
    except BaseException as error:
        failure = dict(stage='closed_samecall', error=repr(error), actual_video=False, quality_gate_clear=False)
        try:
            with (BASE / sys.argv[1] / 'PROBABILITY_OWNER_FINISH_FAILURE.json').open('x', encoding='utf-8') as stream:
                json.dump(failure, stream, indent=2)
        except BaseException:
            pass  # 元の検証例外を保存失敗で置き換えない。
        raise
    assert before == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    with (BASE / sys.argv[1] / 'PROBABILITY_OWNER_FINISH.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
