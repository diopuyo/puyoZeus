"""修正版の原105更新接続回帰。旧fixture・整数復帰の合格条件は変えない。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import proof_cpu_basis as OLD
import inflight_loader_v2 as LOADER


def main() -> int:
    root = LOADER.ROOT
    basis = root.parent / 'g2_reset_settled_basis_gate_2026-09-11_v1'
    paths = [root / n for n in ('quarantine_v2.py', 'inflight_loader_v2.py',
                               'proof_cpu_basis_v3.py')]
    paths += [basis / n for n in ('gate_v3.py', 'actual_connection_v2.py')]
    sha = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = sha()
    OLD.R.C = LOADER.CONNECTION
    OLD.basis_connection = LOADER.basis_connection
    OLD.R.Context = OLD.Context
    code = OLD.R.main()
    unchanged = before == sha()
    output = OLD.R.P.R.Q.ROOT / sys.argv[1]
    with (output / 'SETTLED_BASIS_V3_SOURCE.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(source=before, unchanged=unchanged, original_exit=code,
                       quality_gate_clear=False), stream, indent=2)
    return code or int(not unchanged)


if __name__ == '__main__':
    raise SystemExit(main())
