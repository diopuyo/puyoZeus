"""検収済み原Sessionを最後まで駆動し、新候補の原票と終了票を保持する。"""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys

assert os.environ.get('G2_TARGET_BASIS_ONLY') == '1', 'basis_fixture_environment_required'
import probe_imports as I
import collector_connector_v3 as C
import runtime_session as S

ROOT = Path(__file__).resolve().parent
NAMES = ('run_joint.py', 'runtime_session.py', 'parent_client_v2.py', 'anchor_v2.py',
         'collector_connector_v2.py', 'collector_connector_v3.py', 'raw_upgrade.py', 'raw_upgrade_v2.py')


def sources() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in NAMES}


def main() -> int:
    before = sources()
    original_start, original_create = I.R.V6.V4.start, I.R.V6.V5.create
    with ExitStack() as stack:
        C.install(stack)
        stack.callback(setattr, I.R.V6.V4, 'start', original_start)
        stack.callback(setattr, I.R.V6.V5, 'create', original_create)
        I.R.V6.V4.start, I.R.V6.V5.create = S.start, S.create
        code = I.R.V6.V5.main()
    restored = I.R.V6.V4.start is original_start and I.R.V6.V5.create is original_create
    unchanged = sources() == before
    assert restored and C.DISPATCH not in sys.modules, 'joint_entry_not_restored'
    if '--preflight' in sys.argv:
        return code or int(not unchanged)
    output = ROOT.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1' / sys.argv[1]
    proof = dict(original_exit=code, sources=before, unchanged=unchanged, entry_restored=restored,
                 actual_video=False, quality_gate_clear=False)
    with (output / 'JOINT_RUNTIME_SOURCE.json').open('x', encoding='utf-8') as stream:
        json.dump(proof, stream, indent=2)
    return code or int(not unchanged)


if __name__ == '__main__':
    raise SystemExit(main())
