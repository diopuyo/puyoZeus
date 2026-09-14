"""旧v56のSession生成点まで限定観測し、評価前に止めて全解除を検査する。"""
from __future__ import annotations
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from typing import Any
assert os.environ.get('G2_TARGET_BASIS_ONLY') == '1', 'basis_fixture_environment_required'
CLIENT_ROOT = Path(__file__).resolve().parent
JOINT_ROOT = CLIENT_ROOT.parent / 'g2_joint_collector_runtime_2026-09-12_v1'
sys.path.insert(0, str(JOINT_ROOT))
import probe_imports as I
import collector_connector_v3 as C
import runtime_session as S
sys.path.insert(0, str(CLIENT_ROOT))
import start_client as NEW

ROOT = Path(__file__).resolve().parent
EXPECTED_FRAME = 34932
STOP = 'joint_real_session_constructed_before_evaluation'


def limit(state: dict, capture: Any, original: Any) -> Any:
    owner, loop = state['joint_capture_stack'], capture.loop
    collect = loop.collect_lean
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if args[2] > EXPECTED_FRAME:
            raise ValueError('joint_session_not_reached_by_expected_frame')
        return collect(*args, **kwargs)
    def restore() -> None:
        assert loop.collect_lean is guarded, 'session_probe_foreign_hook'
        loop.collect_lean = collect
    owner.callback(restore)
    loop.collect_lean = guarded
    return original(state, capture)


def main() -> None:
    result: dict[str, Any] = dict(actual_video=False, quality_gate_clear=False, model_inference_performed=False)
    refs: dict[str, Any] = {}
    assert 'anchor_v2' not in sys.modules, 'legacy_anchor_name_preoccupied'
    with ExitStack() as stack:
        original_client = S.CLIENT
        stack.callback(setattr, S, 'CLIENT', original_client)
        S.CLIENT = NEW
        assert Path(NEW.__file__).resolve() == ROOT / 'start_client.py'
        C.install(stack)
        original_final, original_create = C.BASE.final_capture, S.create
        stack.callback(setattr, C.BASE, 'final_capture', original_final)
        stack.callback(setattr, S, 'create', original_create)
        C.BASE.final_capture = lambda state, capture: limit(state, capture, original_final)
        def create(owner: Any, context: dict) -> Any:
            session = original_create(owner, context)
            state = context['state']
            capture, client = state['joint_producer_capture'], state['joint_parent_client']
            assert type(client) is NEW.Client and S.CLIENT is NEW
            assert NEW.T is original_client.T
            assert NEW.worker_path() == ROOT / 'start_worker.py'
            result.update(client_module=type(client).__module__, client_source=str(Path(NEW.__file__).resolve()),
                          worker_path=str(NEW.worker_path()), transport_same=True)
            refs.update(capture=capture, client=client, session=session)
            result.update(frame=capture.last, update_count=capture.count,
                owner_same=owner is state['joint_capture_stack'] is context['stack'],
                child_ready=client.child.poll() is None, source_id=capture.identity['source_id'])
            raise RuntimeError(STOP)
        S.create = create
        I.R.V6.V4.start = S.start
        I.R.V6.V5.create = create
        code = I.R.V6.V5.main()
    if refs:
        result.update(capture_closed=refs['capture'].closed, child_closed=refs['client'].closed,
                      child_exit=refs['client'].child.returncode,
                      witness_closed=refs['session'].witness.closed, evidence_closed=refs['session'].evidence.closed)
    result.update(original_exit=code, planned_stop=STOP)
    with (ROOT / 'SESSION_CONSTRUCTOR_PROBE_v2.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    assert S.CLIENT is original_client, 'client_restoration'
    assert result.get('child_ready') and result.get('transport_same')
    assert code == 1 and result.get('frame') == EXPECTED_FRAME and result.get('owner_same')
    assert result['capture_closed'] and result['child_closed'] and result['child_exit'] == 0
    assert result['witness_closed'] and result['evidence_closed']
    print(json.dumps(result))


if __name__ == '__main__':
    main()
