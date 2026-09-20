"""v42失敗をPuyoGroup全値比較だけで修復する別run。旧資産は不変更。"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('_g2_tracker_snapshot_owned_connection', ROOT / 'connection.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
CPLAN = ROOT.parent / 'g2_cascade_connection_plan_2026-09-11_v1'
sys.path.insert(0, str(CPLAN))
import proof_cascade as OLD


def configure(original: Any) -> Any:
    result = original()
    import side_actual_connection as S
    import cascade_finish as F
    hidden = result.OLD.BASE.H
    hidden.Context = C.derived(hidden.Context, S)
    old_live = F.live
    def live(kept: Any) -> Any:
        report = old_live(kept)
        C.save(kept['state']['output'], kept['state'])
        return report
    F.live = live
    return result


def main() -> int:
    sources = tuple(ROOT.glob('*.py'))
    def digest() -> Any:
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    before, started = digest(), time.monotonic()
    original = OLD.configure
    OLD.configure = lambda: configure(original)
    code = OLD.main()
    output = ROOT.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1' / sys.argv[1]
    with (output / 'SNAPSHOT_REPAIR_SOURCE.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(original_exit=code, unchanged=before == digest(), source=before,
                       seconds=time.monotonic()-started, G2=False), stream, indent=2)
    return code or int(before != digest())


if __name__ == '__main__':
    raise SystemExit(main())
