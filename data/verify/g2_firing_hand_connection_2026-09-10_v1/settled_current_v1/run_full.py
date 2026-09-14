"""専用精算current→次NEXT→通常currentを原83更新へ結合する。"""
from __future__ import annotations
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
PARENT, VERIFY = ROOT.parent, ROOT.parent.parent
sys.path[:0] = [str(ROOT), str(PARENT), str(VERIFY/'g2_firing_continuation_independent_2026-09-10_v1'),
               str(VERIFY/'g2_settled_current_revision_2026-09-10_v1')]
from continuation_v1 import run as R, single_head_v2 as T
import next_hand_v2 as N
import settled_connection as C


def drive(*args: Any) -> Any:
    state, factory = args[3], args[7]
    control = factory.controller
    p = type(control).prepared.__globals__['V1'].P
    receiver = state['postcommit_current_receiver']
    pub = type(receiver).complete.__globals__['T']
    def references(factory: Any, pipe: Any) -> Any:
        return R.G.references(factory, pipe)+(type(control).hold_transition, p.committed, p.proof, p.recover, pub.issue)
    facade = SimpleNamespace(**(vars(R.G) | {'references': references}))
    def install(stack: Any, owner: Any, patch: Any, rows: Any) -> None:
        revision = R.G.R.K.load('_settled_current_revision',
            VERIFY/'g2_settled_current_revision_2026-09-10_v1/recover.py', stack)
        C.install(stack, owner, state, patch, revision.recover_settled_current, rows, N)
    return FunctionType(R.drive.__code__, dict(vars(R), G=facade, T=T, N=SimpleNamespace(install=install)))(*args)


def accepted(output: Path) -> dict[str, Any]:
    execution = R.G.R.K.read(output/'EXECUTION.json')
    assert execution['returned'] and execution['error'] is None, 'settled_full_outer_failed'
    assert execution['builder_restored'] and execution['qualified_restored'] and execution['input_frames_restored']
    data, rows = R.G.R.K.read(output/'DIAGNOSTIC.json'), R.G.R.K.read(output/'FIRING.json')
    assert data['primary'] is None and len(data['completed']) == len(R.I.FRAMES)
    assert data['native_counter_unchanged'] and data['firing_references_restored']
    stages = {name:[row for row in rows if row['stage'] == name] for name in
              ('actual_raw_settled','settled_current_published','next_action_started')}
    assert all(len(value) == 1 for value in stages.values())
    settled, published, started = [value[0] for value in stages.values()]
    assert settled['frame'] <= published['frame'] < started['frame'] and published['consumed'] is False
    assert len(settled['private_state']['consumed_ids']) == 1
    last = data['trace'][-1]
    assert last['history_count'] == 6 and sum(last['inventory']) == 5
    assert last['current_available'] and data['publication']['issued'] == 5
    assert sum(c != 0 for row in last['confirmed'] for c in row) == 5
    return dict(completed_updates=len(data['completed']), settlement_frame=settled['frame'],
        settled_current_frame=published['frame'], next_started_frame=started['frame'],
        next_source_frame=started['source_frame'], early_next=R.EARLY_NEXT,
        live_video_verified=False, next_long_repair_run='NO-GO_PENDING_INDEPENDENT_AND_OTHER_BOUNDARIES')


def guards() -> dict[str, str]:
    paths = list(ROOT.glob('*.py'))
    paths.append(VERIFY/'g2_firing_continuation_independent_2026-09-10_v1/next_hand_v2.py')
    paths.extend(VERIFY/'g2_settled_current_revision_2026-09-10_v1'/name
                 for name in ('recover.py','settled_evidence.py','settled_fixed.py'))
    return R.guards() | {str(path):R.G.R.K.sha(path) for path in paths}


def main() -> int:
    execute = FunctionType(R.execute.__code__, dict(vars(R), drive=drive))
    common = SimpleNamespace(**(vars(R.G.R.K) | {'guards': guards}))
    return FunctionType(R.G.R.main.__code__, dict(vars(R.G.R), K=common, ROOT=ROOT,
        execute=execute, accepted=accepted))()


if __name__ == '__main__':
    raise SystemExit(main())
