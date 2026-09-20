"""小さな原変換CPUを一回実行し、拒否と未閉鎖を分けて保存する。"""
from __future__ import annotations
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import unittest

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
PARENT = ROOT.parent/'g2_firing_hand_connection_2026-09-10_v1'
FIXED = {PARENT/'firing_ticket_v4.py': 'e86213693eff50808ffb9a08ba5f8682a7979797440c342f1c69ce77b24de1b7',
    PARENT/'run_registered_v5.py': 'c2be8b2dce1e59410e4fc9c253271ad434fc074b35d6c25f5a19712e46d16bfa',
    PROJECT/'scripts/next_enqueue_live_shadow_v1.py': 'e5ebff6827119c616319b4598fce1428c98643b621ff02e985736b43783d9237',
    PROJECT/'.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/src/recognition_pipeline.py':
        '6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def guards() -> dict[str, str]:
    for path, expected in FIXED.items(): assert sha(path) == expected, str(path)
    paths = list(FIXED) + [PARENT/name for name in ('firing_ticket_v3.py', 'firing_ticket_v2.py',
        'firing_ticket.py', 'front_probe.py')] + list(ROOT.glob('*.py')) + [ROOT/'PLAN.md',
        ROOT.parent/'g2_firing_policy_2026-09-10_v1/firing_fixed.py',
        PROJECT/'scripts/next_enqueue_freshness_shadow_v1.py',
        PROJECT/'scripts/initial_placement_accounting_shadow_v1.py']
    return {str(path): sha(path) for path in paths}


def style() -> None:
    for path in ROOT.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.FunctionDef):
                assert node.end_lineno-node.lineno+1 <= 50, (path, node.name)
                assert node.returns is not None, (path, node.name)
                assert all(arg.annotation is not None or arg.arg == 'self'
                    for arg in node.args.args + node.args.kwonlyargs)


def main() -> int:
    output = ROOT/sys.argv[1]
    assert output.parent == ROOT
    output.mkdir(exist_ok=False)
    before, started, previous = guards(), time.perf_counter(), sys.getprofile()
    style()
    import test_transform as T
    original = T.Q.V3.V2.OLD.qualified
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(T))
    after = guards()
    unchanged = after == before and previous is sys.getprofile() and T.Q.V3.V2.OLD.qualified is original
    write(output/'GUARDS.json', dict(before=before, after=after, equal=before == after))
    write(output/'CASES.json', T.CASES)
    write(output/'TESTS.json', dict(log=stream.getvalue(), errors=[(str(t), e) for t, e in result.errors],
        failures=[(str(t), e) for t, e in result.failures]))
    value = dict(pid=os.getpid(), seconds=time.perf_counter()-started, tests=result.testsRun,
        passed=result.wasSuccessful(), unchanged=unchanged, imported_torch='torch' in sys.modules,
        foreign_journal_controller='unclosed_entry_binding', full=False, native_pop=False)
    write(output/'RESULT.json', value)
    write(output/'INDEX.json', {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(value), flush=True)
    return 0 if result.wasSuccessful() and unchanged else 1


if __name__ == '__main__':
    raise SystemExit(main())
