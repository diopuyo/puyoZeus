"""新選択型で元Session/元親clientを生成・終了する。入力/producerは人工、採録前停止。"""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import sys
import os
from types import SimpleNamespace as N

import late_loader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_m1_ram_runtime_2026-09-13_v15'))
import owned_adapter as R

STOP = 'constructor_probe_before_any_evaluation'


def fixture(parts: object, load: object, output: object, stack: ExitStack) -> object:
    sys.path.insert(0, str(R.A.BASE))
    import continuous_fixture as F
    f = F.setup(parts, load, output)
    stack.callback(f.mode.stream.close)
    f.j.pipe = f.pipe
    f.j.stream = stack.enter_context((output / 'atomic_journal.jsonl').open('x'))
    f.connection.recovery.provider.journal = f.j
    factory = N(provider=f.connection.recovery.provider)
    registry = parts.binding.OLD.R.Registry(factory)
    binding = registry.bind(factory, f.initial, 'step:10')
    c = parts.binding.Connection.__new__(parts.binding.Connection)
    vars(c).update(vars(f.connection), registry=registry, binding=binding)
    c.recovery.factory = factory
    f.mode.connection, f.mode.native.connection = c, c
    f.state.update(probabilistic_basis_connection=c, private_suffix_factory=factory,
        joint_capture_stack=stack, joint_producer_capture=N(identity=dict(source_id='source', run_id='run')))
    return N(state=f.state, pipe=f.pipe, factory=factory)


def settings(pid: int) -> dict:
    keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    environment = dict(v.split('=', 1) for v in Path(f'/proc/{pid}/environ').read_text().split('\0') if '=' in v)
    assert all(environment[k] == '2' for k in keys), 'normal_child_threads'
    affinity = sorted(os.sched_getaffinity(pid))
    assert len(affinity) > 1 and affinity == sorted(os.sched_getaffinity(0)), 'normal_child_affinity'
    return dict(threads={k: environment[k] for k in keys}, affinity=affinity, forced_frame_sleep=False)


def main() -> None:
    output = ROOT / 'old_session_constructor_v1'
    output.mkdir(exist_ok=False)
    refs = {}
    with ExitStack() as outer:
        R.configured(outer)
        import probe_native_merge
        owner = sys.modules[R.A.A.A.V4.OWNED_ALIAS]
        load = owner.bootstrap().load
        create = R.A.A.A.V4.A.D.session_creator(load, (35370, 35410))
        parts = owner.dependencies().modules()
        load = owner.bootstrap().load
        try:
            with ExitStack() as inner:
                f = fixture(parts, load, output, inner)
                context = dict(state=f.state, pipe=f.pipe, factory=f.factory, stack=inner)
                session = create(inner, context)
                thermal = settings(f.state['joint_parent_client'].child.pid)
                client = f.state['joint_parent_client']
                refs.update(session=session, client=client)
                assert client.child.poll() is None
                assert Path(type(session).completed.__code__.co_filename).name == 'live_session.py'
                assert not hasattr(session, 'schedule_ready')
                raise RuntimeError(STOP)
        except RuntimeError as error:
            assert str(error) == STOP
        session, client = refs['session'], refs['client']
        assert client.closed and client.child.returncode == 0
        assert session.witness.closed and session.evidence.closed
        assert not session.saved
    report = dict(original_Session_constructor=True, original_parent_client_ready_and_closed=True,
        child_exit=client.child.returncode, thermal=thermal, old_session_confirmed=True,
        actual_completed_source=type(session).completed.__code__.co_filename, witness_closed=True,
        artificial_initial_basis_pipe_and_producer=True, original_first_basis_acquisition=False,
        creator_before_modules=True, actual_updates=0, actual_save=False, actual_video=False, planned_stop=STOP, quality_gate_clear=False)
    with (output / 'RESULT.json').open('x') as stream: json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
