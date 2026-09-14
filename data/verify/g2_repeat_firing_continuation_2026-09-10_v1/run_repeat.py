"""既存83更新接続を再用し、二回目発火まで110更新へ限定延長する。"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
PARENT = VERIFY/'g2_firing_hand_connection_2026-09-10_v1'
sys.path[:0] = [str(PARENT/'settled_current_v1'), str(PARENT),
               str(VERIFY/'g2_firing_ticket_handoff_2026-09-10_v1')]
import run_full as Q
import settled_connection_v2 as C
import repeat_inputs as I

MAX_UPDATES = 120


def drive(*args: Any) -> Any:
    with ExitStack() as stack:
        root = VERIFY/'g2_firing_ticket_handoff_2026-09-10_v1'
        load = Q.R.G.R.K.load
        rollover = load('_repeat_firing_handoff', root/'handoff.py', stack)
        connection = Q.R.G.F
        original = connection.installed
        ticket = SimpleNamespace(**(vars(connection.F) | {'install': rollover.install}))
        installed = contextmanager(FunctionType(original.__wrapped__.__code__,
            dict(original.__wrapped__.__globals__, F=ticket)))
        Q.R.patch(stack, connection, 'installed', installed)
        return FunctionType(Q.drive.__code__, dict(vars(Q), C=C))(*args)


def accepted(output: Path) -> dict[str, Any]:
    execution = Q.R.G.R.K.read(output/'EXECUTION.json')
    assert execution['returned'] and execution['error'] is None, 'repeat_outer_failed'
    assert execution['builder_restored'] and execution['qualified_restored'] and execution['input_frames_restored']
    data, rows = Q.R.G.R.K.read(output/'DIAGNOSTIC.json'), Q.R.G.R.K.read(output/'FIRING.json')
    assert data['primary'] is None and len(data['completed']) == len(I.FRAMES)
    assert data['native_counter_unchanged'] and data['firing_references_restored']
    settled = [r for r in rows if r['stage'] == 'actual_raw_settled']
    published = [r for r in rows if r['stage'] == 'settled_current_published']
    registered = [r for r in rows if r['stage'] == 'firing_registered']
    assert len(settled) == len(published) == len(registered) == 2
    assert len({r['origin']['origin_id'] for r in registered}) == 2
    assert all(a['frame'] <= b['frame'] and b['consumed'] is False for a,b in zip(settled,published))
    assert len(settled[-1]['private_state']['origins']) == len(settled[-1]['private_state']['consumed_ids']) == 2
    last = data['trace'][-1]
    assert last['history_count'] == 9 and sum(last['inventory']) == 4
    assert data['publication']['issued'] == 7
    assert sum(c != 0 for row in last['confirmed'] for c in row) == 4
    return dict(completed_updates=len(data['completed']), fires=2,
        registered_frames=[r['frame'] for r in registered], settled_frames=[r['frame'] for r in settled],
        current_frames=[r['frame'] for r in published], old_origins_retained=True,
        artificial=True, live_video_verified=False, next_long_repair_run='NO-GO_PENDING_OTHER_BOUNDARIES')


def guards() -> dict[str, str]:
    paths = [*ROOT.glob('*.py'), ROOT/'PLAN.md']
    paths.extend(VERIFY/'g2_firing_ticket_handoff_2026-09-10_v1'/name
                 for name in ('firing_handoff_fixed.py','firing_handoff_lifecycle.py','handoff.py'))
    return Q.guards() | {str(path):Q.R.G.R.K.sha(path) for path in paths}


def main() -> int:
    execute = FunctionType(Q.R.execute.__code__, dict(vars(Q.R), I=I, MAX_UPDATES=MAX_UPDATES, drive=drive))
    common = SimpleNamespace(**(vars(Q.R.G.R.K) | {'guards': guards}))
    return FunctionType(Q.R.G.R.main.__code__, dict(vars(Q.R.G.R), K=common, ROOT=ROOT,
        execute=execute, accepted=accepted))()


if __name__ == '__main__':
    raise SystemExit(main())
