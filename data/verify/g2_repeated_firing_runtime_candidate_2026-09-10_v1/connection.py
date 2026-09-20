"""人工入力を持たず、検収済みの反復連鎖部品を実Factoryへ設置する。"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'g2_repeat_firing_continuation_2026-09-10_v1'))
import run_repeat as P

G, K = P.Q.R.G, P.Q.R.G.R.K


def references(factory: Any, pipe: Any, state: Any) -> tuple[Any, ...]:
    control = factory.controller
    policy = type(control).prepared.__globals__['V1'].P
    pub = type(state['postcommit_current_receiver']).complete.__globals__['T']
    return G.references(factory, pipe)+(type(control).hand, type(control).observed,
        type(control).hold_transition, policy.committed, policy.proof, policy.recover, pub.issue)


def install(stack: Any, factory: Any, pipe: Any, state: Any, rows: list[Any]) -> None:
    assert not factory.controller.history, 'repeat_install_after_baseline'
    base = K.load('_repeat_runtime_deferred', G.DEFERRED/'connection.py', stack)
    identity = K.load('_repeat_runtime_identity', G.DEFERRED/'connection_v2.py', stack)
    rollover = K.load('_repeat_runtime_handoff',
        ROOT.parent/'g2_firing_ticket_handoff_2026-09-10_v1/handoff.py', stack)
    base.patch(stack, base.D, 'check', G.H2.check)
    configured = P.Q.T.configure(stack, base, factory.controller)
    stack.enter_context(identity.installed(configured, factory.controller))
    ticket = SimpleNamespace(**(vars(G.F.F) | {'install': rollover.install}))
    body = G.F.installed.__wrapped__
    firing = contextmanager(FunctionType(body.__code__, dict(body.__globals__, F=ticket)))
    stack.enter_context(firing(factory, pipe, configured, G.P, rows))
    P.Q.R.S.install(stack, factory.controller, base.patch, rows)
    revision = K.load('_repeat_runtime_revision',
        ROOT.parent/'g2_settled_current_revision_2026-09-10_v1/recover.py', stack)
    P.C.install(stack, factory.controller, state, base.patch,
        revision.recover_settled_current, rows, P.Q.N)


def guards() -> dict[str, str]:
    return P.guards() | {str(path): K.sha(path) for path in (*ROOT.glob('*.py'), ROOT/'PLAN.md')}
