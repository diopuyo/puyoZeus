"""元入口のanchor/finishだけを差替え、新私有コードの不変も検査する。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_hidden_basis_initialization_2026-09-11_v1'))
import proof_cpu_finalized as OLD
import anchor_v2 as A
import finish_v2 as F


def main() -> int:
    paths = list(ROOT.glob('*.py'))
    digest = lambda: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = digest()
    OLD.A, OLD.F = A, F
    code = OLD.main()
    output = OLD.BASE.H.OLD.DRIVER.P.R.Q.ROOT / sys.argv[1]
    unchanged = before == digest()
    with (output / 'ARCHIVE_LIFETIME_SOURCE.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(source=before, unchanged=unchanged, original_exit=code,
                       quality_gate_clear=False), stream, indent=2)
    return code or int(not unchanged)


if __name__ == '__main__':
    raise SystemExit(main())
