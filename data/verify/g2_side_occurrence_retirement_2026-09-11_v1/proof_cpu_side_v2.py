"""v36親GOを再用し、片側reset→occurrence退役→原105更新を一括検収する。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
SIDE = ROOT.parent / 'g2_side_actual_run_2026-09-11_v1'
sys.path.insert(0, str(SIDE))
import predecessor as G


def validate(output: Path) -> None:
    retirement = json.loads((output / 'SIDE_OCCURRENCE_RETIREMENT.json').read_bytes())
    side = json.loads((output / 'SIDE_ACTUAL_CONNECTION.json').read_bytes())
    assert retirement['stage'] == 'side_occurrence_retired' and retirement['other_sides_unchanged'] is True
    assert retirement['new_epoch'] == retirement['old_epoch'] + 1 and retirement['absent_noop'] is False
    assert retirement['new']['epoch'] == retirement['new_epoch'] and retirement['new']['blocked'] is None
    assert side['stage'] == 'side_reset_connected' and side['other_side_unchanged'] is True
    assert side['full_reset_calls'] == 0 and side['frame'] == retirement['frame']


def main() -> int:
    approval = G.verify()
    paths = [p for root in (ROOT, SIDE) for p in root.glob('*.py')]
    digest = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before, started = digest(), time.monotonic()
    sys.path[:0] = [str(ROOT.parent / 'g2_archive_close_boundary_2026-09-11_v1'),
                   str(ROOT.parent / 'g2_hidden_basis_initialization_2026-09-11_v1')]
    import proof_cpu_finalized_v2 as F
    import side_actual_connection as S
    import retirement as R
    F.OLD.BASE.H.Context = R.derived(S.derived(F.OLD.BASE.H.Context))
    output = F.OLD.BASE.H.OLD.DRIVER.P.R.Q.ROOT / sys.argv[1]
    failure = None
    try:
        code = F.main()
        if code == 0: validate(output)
    except Exception:
        code, failure = 1, traceback.format_exc()
    unchanged = before == digest()
    with (output / 'SIDE_RETIREMENT_SOURCE.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(source=before, unchanged=unchanged, original_exit=code, error=failure,
            seconds=time.monotonic() - started, predecessor=approval, quality_gate_clear=False), stream, indent=2)
    return code or int(not unchanged)


if __name__ == '__main__':
    raise SystemExit(main())
