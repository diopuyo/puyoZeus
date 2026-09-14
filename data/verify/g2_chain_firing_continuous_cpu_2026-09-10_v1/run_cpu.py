"""凍結 common/session を再用する排他 CPU 診断入口。"""
from __future__ import annotations
from contextlib import ExitStack
import ast
import os
from pathlib import Path
import sys
import time
import traceback
from types import FunctionType
from typing import Any
import artificial_inputs as I
import full_driver as D

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
RUNTIME = VERIFY/'g2_history_publication_probe_runtime_2026-09-10_v11'
PRIOR = VERIFY/'g2_history_baseline_adoption_independent_2026-09-10_v1'
TARGET = PRIOR/'target_current_full_v1/check_current.py'
BASE = PRIOR/'target_full_v1/check_full_v2.py'
sys.path.insert(0, str(RUNTIME))
import common as K
import cpu_smoke as S


def execute(output: Path) -> Any:
    with ExitStack() as stack:
        target = K.load('_firing_target_current', TARGET, stack)
        assert target.K is K and S.K is K
        original_session = S.A.session
        def drive(q: Any, *args: Any) -> Any:
            assert q.K is K and q.S is S and S.A.session is original_session
            return D.drive(q, target, *args)
        return FunctionType(target.execute.__code__, dict(vars(target), FRAMES=I.FRAMES, drive=drive))(output)


def accepted(output: Path) -> dict[str, Any]:
    data = K.read(output/'DIAGNOSTIC.json')
    assert data['primary']['message'] == 'history_origin_unknown'
    assert data['native_counter_unchanged'] and data['score_input_references_restored']
    assert data['raw_input_references_restored'] and data['raw']['closed'] and data['sink']['closed']
    rows = {r['frame']: r for r in data['trace']}
    for frame, count in ((I.GY_END, 1), (I.BP_END, 2), (I.PP_END, 3)):
        assert rows[frame]['history_count'] == count and sum(rows[frame]['inventory']) == count*2
        assert rows[frame]['current_available'] is True
    last = data['trace'][-1]
    assert last['active']['mechanism'] == 'formula_read' and last['active']['before_board'] == data['control']['before']
    assert last['history_count'] == 3 and data['publication']['issued'] == 3
    assert any(r['body'] == 'no_origin' and r.get('returned') is False for r in data['origin_bodies'])
    return dict(counterexample_confirmed=True, known_unresolved=True, next_long_repair_run='NO-GO',
        completed_updates=len(data['completed']), failed_frame=data['primary']['frame'],
        original_processing_exit=1, primary=data['primary'], native_counter_unchanged=True,
        current_events=[I.GY_END, I.BP_END, I.PP_END], firing_hand_in_origin=False)


def main() -> int:
    output = ROOT/sys.argv[1]
    assert output.parent == ROOT and output.name.startswith('cpu_v') and len(I.FRAMES) <= 40
    output.mkdir(exist_ok=False)
    paths = (TARGET, BASE, *ROOT.glob('*.py'), ROOT/'PLAN.md')
    guards = K.guards() | {str(p): K.sha(p) for p in paths}
    started, code, error = time.perf_counter(), 0, None
    try:
        import cv2
        import torch
        cv2.setNumThreads(2)
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        try:
            execute(output)
        except BaseException:
            error = traceback.format_exc()
        result = accepted(output)
    except BaseException:
        result, code = dict(diagnostic_error=traceback.format_exc()), 1
    unchanged = all(K.sha(Path(p)) == digest for p, digest in guards.items())
    assert unchanged
    for p in ROOT.glob('*.py'):
        nodes = [n for n in ast.walk(ast.parse(p.read_bytes())) if isinstance(n, ast.FunctionDef)]
        assert all(n.returns is not None and n.end_lineno-n.lineno+1 <= 50 for n in nodes)
    result.update(pid=os.getpid(), seconds=time.perf_counter()-started, exit_code=code,
        outer_exception=error, guards=guards, guards_unchanged=unchanged,
        cuda_initialized=torch.cuda.is_initialized(), profile_restored=sys.getprofile() is None,
        physical_certified=False)
    K.write(output/'RESULT.json', result)
    K.write(output/'INDEX.json', {str(p.relative_to(output)): K.sha(p) for p in output.rglob('*') if p.is_file()})
    print({k: result[k] for k in ('pid', 'seconds', 'exit_code', 'diagnostic_error') if k in result}, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
