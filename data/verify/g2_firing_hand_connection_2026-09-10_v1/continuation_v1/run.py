"""既存39更新の後に人工消去と次手を追加する有限83更新CPU。"""
from __future__ import annotations
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys
import traceback
from types import FunctionType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import run_registered as G
import firing_ticket_v5 as Q
from continuation_v1 import inputs as I, next_hand as N, settlement as S, transforms as T

MAX_UPDATES = 90
EARLY_NEXT = '--early-next' in sys.argv


def drive(q: Any, target: Any, m: Any, state: Any, real: Any, fixture: Any,
          binding: Any, factory: Any) -> dict[str, Any]:
    pipe, rows = real[0], []
    counters = {s:(getattr(pipe, '_tsumo_count_'+s), deepcopy(getattr(pipe, '_tsumo_count_'+s))) for s in ('1p','2p')}
    data: dict[str, Any] = dict(trace=[], completed=[], primary=None)
    refs = G.references(factory, pipe)+(type(factory.controller).hand, type(factory.controller).observed)
    try:
        with ExitStack() as stack:
            base = G.R.K.load('_firing_deferred_base', G.DEFERRED/'connection.py', stack)
            identity = G.R.K.load('_firing_deferred_identity', G.DEFERRED/'connection_v2.py', stack)
            base.patch(stack, base.D, 'check', G.H2.check)
            configured = T.configure(stack, base, factory.controller)
            stack.enter_context(identity.installed(configured, factory.controller))
            stack.enter_context(G.F.installed(factory, pipe, configured, G.P, rows))
            S.install(stack, factory.controller, base.patch, rows)
            N.install(stack, factory.controller, base.patch, rows)
            G.R.D.loop(q, target, m, state, real, fixture, binding, factory, data)
        return dict(firing_continuation_reached=True, physical_certified=False)
    finally:
        data['firing_references_restored'] = refs == G.references(factory, pipe)+(type(factory.controller).hand, type(factory.controller).observed)
        data['native_counter_unchanged'] = all(getattr(pipe, '_tsumo_count_'+s) is ref and ref == old
            for s,(ref,old) in counters.items())
        receiver = state['postcommit_current_receiver']
        data['publication'] = dict(issued=receiver.issued, released=receiver.released, errors=receiver.errors)
        q.K.write(state['output']/'DIAGNOSTIC.json', fixture.subject.base.json_value(data))
        q.K.write(state['output']/'FIRING.json', fixture.subject.base.json_value(rows))


def patch(stack: Any, obj: Any, name: str, value: Any) -> None:
    old = getattr(obj, name)
    stack.callback(setattr, obj, name, old)
    setattr(obj, name, value)


def execute(output: Path) -> Any:
    evidence: dict[str, Any] = dict(returned=False, error=None, early_next=EARLY_NEXT, planned_updates=len(I.FRAMES))
    assert len(I.FRAMES) <= MAX_UPDATES
    def driven(*args: Any) -> Any:
        with Q.installed(args[4][0], evidence):
            return drive(*args)
    try:
        with ExitStack() as stack:
            I.install(stack, G.R.I, patch, early_next=EARLY_NEXT)
            result = FunctionType(G.execute.__code__, dict(vars(G), drive=driven))(output)
        evidence.update(returned=True, builder_restored=result['builder_restored'], closed=result['closed'])
        return result
    except BaseException:
        evidence['error'] = traceback.format_exc()
        raise
    finally:
        evidence['input_frames_restored'] = len(G.R.I.FRAMES) == 39
        G.R.K.write(output/'EXECUTION.json', evidence)


def accepted(output: Path) -> Any:
    execution = G.R.K.read(output/'EXECUTION.json')
    assert execution['returned'] and execution['error'] is None, 'continuation_outer_failed'
    assert execution['builder_restored'] and execution['qualified_restored'] and execution['input_frames_restored']
    data, rows = G.R.K.read(output/'DIAGNOSTIC.json'), G.R.K.read(output/'FIRING.json')
    assert data['primary'] is None and len(data['completed']) == len(I.FRAMES)
    assert data['native_counter_unchanged'] and data['firing_references_restored']
    settled = [row for row in rows if row['stage'] == 'actual_raw_settled']
    started = [row for row in rows if row['stage'] == 'next_action_started']
    assert len(settled) == len(started) == 1 and settled[0]['frame'] <= started[0]['frame']
    assert len(settled[0]['private_state']['consumed_ids']) == 1
    last = data['trace'][-1]
    assert last['history_count'] == 6 and sum(last['inventory']) == 5
    assert last['current_available'] and data['publication']['issued'] == 4
    assert sum(c != 0 for row in last['confirmed'] for c in row) == 5
    return dict(completed_updates=len(data['completed']), settlement_frame=settled[0]['frame'],
        next_started_frame=started[0]['frame'], next_source_frame=started[0]['source_frame'],
        early_next=EARLY_NEXT, origin_history_retained=True, actual_input_artificial=True,
        live_video_verified=False, next_long_repair_run='NO-GO_PENDING_INDEPENDENT_AND_OTHER_BOUNDARIES')


def guards() -> dict[str, str]:
    paths = [*G.ROOT.glob('*.py'), *ROOT.glob('*.py')]
    return G.guards() | {str(path): G.R.K.sha(path) for path in paths}


def main() -> int:
    common = SimpleNamespace(**(vars(G.R.K) | {'guards': guards}))
    return FunctionType(G.R.main.__code__, dict(vars(G.R), K=common, ROOT=ROOT,
        execute=execute, accepted=accepted))()


if __name__ == '__main__':
    raise SystemExit(main())
