"""原fixtureの最初の一更新だけで公開observerの実装着/通過を観測する。意図的停止。"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
REPAIR = ROOT.parent / 'g2_normal_hand_basis_count_repair_2026-09-11_v1'
OBSERVER = ROOT.parent / 'g2_provisional_context_capture_2026-09-09_v1/observer.py'
DRIVE_SUFFIX = '/tests/test_diagnose_video38_next_enqueue_live_shadow_v1.py'
STOP = 'limited_observer_probe_after_one_original_update'
ROWS: list[dict[str, Any]] = []
DEADLINE = 0.0


def profile(frame: Any, event: str, value: Any) -> None:
    if time.monotonic() > DEADLINE:
        raise RuntimeError('limited_observer_probe_timeout')
    path = frame.f_code.co_filename.replace('\\', '/')
    name = frame.f_code.co_name
    selected = path == OBSERVER.as_posix() and name in ('__init__', 'wrapped')
    drive = (path.endswith(DRIVE_SUFFIX) and name == 'drive') or name == 'collect_lean'
    if not selected and not drive:
        return  # 無関係なimport/comprehensionのf_localsは触らない。
    local = frame.f_locals
    if path == OBSERVER.as_posix():
        if name == '__init__' and event == 'return':
            rec = local['self']
            expected = getattr(rec, 'expected', [])
            ROWS.append(dict(kind='context_constructed', expected_count=len(expected),
                first=expected[0] if expected else None, last=expected[-1] if expected else None,
                module=type(rec).__module__, source_file=path))
        if name == 'wrapped' and event == 'call' and 'frame_idx' in local:
            rec = local.get('rec')
            ROWS.append(dict(kind='context_update_called', frame=local['frame_idx'],
                in_expected=local['frame_idx'] in rec.expected if rec is not None else None))
    if drive and event == 'return':
        ROWS.append(dict(kind='original_drive_return', frame=local.get('frame'),
                         returned_frame=getattr(value, 'frame_idx', None), source_file=path,
                         function=name, local_keys=sorted(local)))
        raise RuntimeError(STOP)


def main() -> None:
    global DEADLINE
    sys.path.insert(0, str(REPAIR))
    spec = importlib.util.spec_from_file_location('_g2_publication_probe_fixed_entry', REPAIR / 'run.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    previous, code = sys.getprofile(), None
    DEADLINE = time.monotonic() + 120.0
    try:
        sys.setprofile(profile)
        code = entry.main()
    finally:
        sys.setprofile(previous)
        with (ROOT / 'OBSERVER_PROBE_v3.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(rows=ROWS, original_exit=code, planned_stop=STOP,
                           actual_video=False, quality_gate_clear=False), stream, indent=2)
    assert code == 1 and len([r for r in ROWS if r['kind'] == 'original_drive_return']) == 1
    print(json.dumps(dict(rows=ROWS, original_exit=code, quality_gate_clear=False)))


if __name__ == '__main__':
    main()
