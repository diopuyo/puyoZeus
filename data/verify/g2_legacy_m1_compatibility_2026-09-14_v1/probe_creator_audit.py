"""原creator戻り値→before_create→原create→実auditまでの人工履歴CPU対照。"""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace as N
from typing import Any

ROOT = Path(__file__).resolve().parent
STOP = RuntimeError('CPU監査到達後の計画終了')
sys.path.insert(0, str(ROOT.parent / 'g2_second_pending_prefix_2026-09-14_v3'))
import cpu_constructor_context as F


def execute(adapter: Any, output: Path) -> dict:
    import src.chain
    owner = sys.modules[adapter.A.A.A.V4.OWNED_ALIAS]
    create = adapter.A.A.A.V4.A.D.session_creator(owner.bootstrap().load, (35370, 35410))
    parts = owner.dependencies().modules()
    with ExitStack() as stack:
        values = F.context(stack, owner.bootstrap().load, parts, output)
        history = F.history(stack, owner.bootstrap().load, values)
        history.sealed = False
        history.stream = stack.enter_context((output / 'EARLY_ORIGIN_HISTORY.jsonl').open('a'))
        adapter.A.A.A.V4.replace_owned(stack, adapter.CURRENT, 'history', history)
        values['state']['joint_producer_capture'] = N(
            identity=dict(source_id='synthetic-creator-source', run_id='synthetic-creator-run'),
            snapshot=lambda *args: (_ for _ in ()).throw(AssertionError('CPUでは採録しない')))
        result = {}
        try:
            with ExitStack() as inner:
                values['state']['joint_capture_stack'] = inner
                values['stack'] = inner
                session = create(inner, values)
                connection = adapter.CURRENT.connection
                assert type(session) is connection.bound['binding']['cls']
                assert connection.receipt['session_class_verified']
                assert (output / 'EARLY_ORIGIN_CONNECTION.json').exists()
                sys.modules['_g2_pub_runtime_live_session'].attach(inner, session)
                result = dict(actual_creator_return_used=True, actual_before_create_and_audit=True,
                    final_mro=[cls.__module__ + '.' + cls.__qualname__ for cls in type(session).__mro__])
                raise STOP
        except RuntimeError as error:
            if error is not STOP:
                raise
        assert session.restored and session.witness.closed and session.evidence.closed
        assert session.second_settled_notice_stream.closed and session.schedule_stream.closed
        result['session_saves_closed'] = True
    assert history.closed
    return result


def main() -> None:
    version, attempt = sys.argv[1:]
    assert version == 'v38' and attempt.replace('_', '').isalnum()
    output = ROOT / ('creator_' + version + '_' + attempt)
    output.mkdir(exist_ok=False)
    sys.path.insert(0, str(ROOT.parent / ('g2_second_prefix_runtime_2026-09-14_' + version)))
    import owned_adapter as A
    sys.path.insert(0, str(ROOT.parent / 'g2_settled_basis_actual_mismatch_2026-09-12_v1'))
    started, error, result = time.perf_counter(), None, {}
    try:
        with ExitStack() as stack:
            selected = A.configured(stack)
            import probe_native_merge
            with selected.__globals__['S'].configured():
                result = execute(A, output)
    except BaseException as failure:
        error = repr(failure)
    value = dict(result=result, error=error, seconds=time.perf_counter()-started,
        artificial_initial_history_and_producer=True, actual_video=False, quality_gate_clear=False,
        source_aliases_released=not any(name in sys.modules for name in A.SC.NAMES))
    with (ROOT / ('CREATOR_' + version + '_' + attempt + '.json')).open('x') as stream:
        json.dump(value, stream, indent=2)
    print(json.dumps(value))
    assert error is None and result['actual_before_create_and_audit'], error


if __name__ == '__main__':
    main()
