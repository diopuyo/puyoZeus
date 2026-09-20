"""新Coreを原105更新へ接続し、評価前の計画停止と解除を限定検査する。"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import asdict
import json
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / 'g2_joint_collector_runtime_2026-09-12_v1'
END, COUNT = 34980, 105
STOP = 'positive_core_105_updates_verified_stop'
assert os.environ.get('G2_TARGET_BASIS_ONLY') == '1', 'basis_fixture_environment_required'
sys.path.insert(0, str(RUN))
import probe_imports as I
import collector_connector_v3 as C
import runtime_session as S
import core_fixture_v2 as F
import constructor_fixture as CFG
import first_static_reflection as FIRST


def positive(boundary: Any, refs: dict) -> dict:
    state, factory = boundary.state, boundary.context['factory']
    context = state['live_empty_reset_context']
    assert isinstance(context, refs['core']) and context.error is None, 'actual_new_core'
    report = state['live_probability_context']
    assert report['stage'] == 'installed' and report['error'] is None
    assert sum(r.get('kind') == 'probabilistic_context_installed' for r in context.rows) == 1
    mode, lease = state['probabilistic_tracking_mode'], state['repeat_scope_guard'].reset_lease
    assert mode.activation is not None and mode.native is not None and mode.error is None
    refs['deferred'].modules().lease.verify(mode, lease)
    lease.archive.verify()
    recovery = context.recovery
    assert recovery.reset_count == 1 and recovery.baseline_count == 0 and recovery.pending is None
    old, saved, record = recovery.archive[0]
    assert old.owner.state is saved and asdict(saved) == record['owner']
    current = mode.connection.registry.current(mode.connection.binding)
    captured = boundary.session.capture()
    assert captured.frame == END and captured.values[0] is current, 'current_evaluation_binding'
    assert not mode.native.pending, 'current_native_pending'
    assert factory.provider.journal.steps == COUNT * 2
    capture, client = state['joint_producer_capture'], state['joint_parent_client']
    assert capture.count == COUNT and capture.last == END and boundary.session is not None
    refs.update(capture=capture, client=client, session=boundary.session, boundary=boundary)
    return dict(frame=END, updates=capture.count, journal_steps=factory.provider.journal.steps,
                actual_core=True, perform_count=1, activation=mode.activation, current_frame=current.frame,
                baseline_count=0, pending_none=True, old_archive_unchanged=True,
                basis_frames=[value.frame for value in captured.values], evaluation_frame=captured.frame,
                journal_tokens=list(captured.tokens), context_digest=captured.digest)


def install(stack: ExitStack, refs: dict, result: dict) -> None:
    if 'palette_fixture' not in sys.modules:
        path = ROOT.parent / 'g2_hidden_current_candidate_2026-09-10_v1/palette_fixture.py'
        spec = importlib.util.spec_from_file_location('palette_fixture', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['palette_fixture'] = module
        stack.callback(sys.modules.pop, 'palette_fixture')
        spec.loader.exec_module(module)
    palette = sys.modules['palette_fixture']
    assert Path(palette.__file__).resolve() == ROOT.parent / 'g2_hidden_current_candidate_2026-09-10_v1/palette_fixture.py'
    CFG.install(stack, palette, result.setdefault('constructor_fixture', {}))
    runtime = I.R.V6.V5.V4.OLD
    original = runtime.configure
    def configure(*args: Any) -> Any:
        value = F.replace(original(*args), refs)
        stack.callback(setattr, refs['hidden'], 'Context', refs['original'])
        return value
    stack.callback(setattr, runtime, 'configure', original)
    runtime.configure = configure
    original_create = S.create
    def create(owner: Any, context: dict) -> Any:
        session = original_create(owner, context)
        refs['first_static'] = FIRST.install(owner, session)
        return session
    stack.callback(setattr, S, 'create', original_create)
    S.create = create
    prior = S.BOUNDARY.Boundary.collect
    def collect(boundary: Any, *args: Any, **kwargs: Any) -> Any:
        frame = args[2]
        assert frame <= END, 'positive_core_frame_limit'
        value = prior(boundary, *args, **kwargs)
        if frame == END:
            try:
                result.update(positive(boundary, refs))
            except BaseException as error:
                result['positive_assertion_error'] = repr(error)
                raise
            result['planned_stop_reached'] = True
            raise RuntimeError(STOP)
        return value
    stack.callback(setattr, S.BOUNDARY.Boundary, 'collect', prior)
    S.BOUNDARY.Boundary.collect = collect
    for module, key, replacement in ((I.R.V6.V4, 'start', S.start), (I.R.V6.V5, 'create', S.create)):
        stack.callback(setattr, module, key, getattr(module, key))
        setattr(module, key, replacement)


def main() -> None:
    refs: dict[str, Any] = {}
    result: dict[str, Any] = dict(actual_video=False, quality_gate_clear=False,
        model_inference_performed=False, artificial_inputs=True, full_finalizer_verified=False)
    with ExitStack() as stack:
        C.install(stack)
        install(stack, refs, result)
        code = I.R.V6.V5.main()
    if '--preflight' in sys.argv:
        assert code == 0 and result['constructor_fixture']['restored']
        print(json.dumps(dict(preflight=True, constructor_selected=False, updates=0, quality_gate_clear=False)))
        return
    result['original_exit'] = code
    if 'session' in refs:
        session, client = refs['session'], refs['client']
        result.update(capture_closed=refs['capture'].closed, child_closed=client.closed,
            child_exit=client.child.returncode, session_restored=session.restored,
            witness_closed=session.witness.closed, evidence_closed=session.evidence.closed,
            boundary_restored=refs['boundary'].restored, boundary_error=repr(refs['boundary'].error))
    if 'first_static' in refs:
        result['first_static_reflection'] = refs['first_static']
    with (ROOT / 'POSITIVE_CORE_PROBE_v3.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    assert result['constructor_fixture']['selected'] and result['constructor_fixture']['restored']
    assert code == 1 and result.get('planned_stop_reached'), 'positive_probe_not_reached'
    assert all(result[k] for k in ('capture_closed', 'child_closed', 'session_restored',
                                  'witness_closed', 'evidence_closed', 'boundary_restored'))
    assert result['child_exit'] == 0 and result['boundary_error'] == 'None'
    assert result['first_static_reflection']['restored'] and result['first_static_reflection']['checks']
    assert result['first_static_reflection']['error'] is None
    assert C.DISPATCH not in sys.modules, 'dispatcher_not_restored'
    print(json.dumps(result))


if __name__ == '__main__':
    main()
