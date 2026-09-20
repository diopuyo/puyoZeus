"""原prefix検収後、同一factoryで原resetと新25更新を連続実行する。"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import asdict
import json
import sys
from typing import Any
import post_inputs as I
import run_qualification as Q
import boundary_runtime as B


def verify(recovery: Any, lease: Any, factory: Any, state: Any, trace: Any, prior: Any) -> Any:
    lease.observe_wait()
    assert recovery.reset_count == recovery.baseline_count == 1 and recovery.pending is None
    assert lease.outer_calls == lease.native_calls == 1 and not lease.waiting
    lease.archive.verify()
    old, saved, record = recovery.archive[0]
    assert old is lease.archive.binding and old.owner.state is saved and asdict(saved) == record['owner']
    new = factory.controller.history['1P']
    assert new.owner is not old.owner and new.scope[2] == old.scope[2]+1
    assert new.owner.state.current is not None and new.owner.state.current.grid == new.current
    published = B.publication(factory, state, recovery)
    receiver, consumer = state['postcommit_current_receiver'], state['postcommit_publication_consumer']
    assert receiver.issued == prior['issued']+1 and receiver.released == prior['released']
    assert not receiver.errors and not consumer.errors
    assert all(not r['changed_sides'] and r['full_before'] == r['full_after'] for r in consumer.rows)
    assert len(trace) == len(I.POST) and trace[-1]['frame'] == I.POST[-1]
    assert factory.provider.journal.steps == (76+len(I.POST))*2
    return dict(reset_executed=True, original_reset_outer=1, original_reset_native=1,
        old_private_counter=sum(saved.counter), old_integer_preserved=True,
        old_integer_coexistence=False, new_owner=True, baseline_count=1,
        waiting_updates=published['waiting_updates'], acquisition_updates=published['acquisition_updates'],
        publication_boundary=published, new_current_count=sum(r['current'] for r in trace),
        issued_delta=receiver.issued-prior['issued'], released_delta=0, consumer_changes=0,
        original_J_steps=factory.provider.journal.steps, multi_scope_finalizer_verified=False,
        artificial_inputs=True, physical_verified=False, quality_gate_clear=False)


def extend(context: Any, result: Any) -> None:
    state, factory, pipe = (context[n] for n in ('state', 'factory', 'pipe'))
    stack, clock, cap = (context[n] for n in ('stack', 'clock', 'cap'))
    proof = Q.KEPT['prefix_evidence']
    guard = state['repeat_scope_guard']
    holder = Q.recovery(factory, pipe, state, guard.reset_lease.evidence)
    module = sys.modules[type(holder).__module__]
    module.I = I.shifted(module.I)
    recovery = module.install(stack, factory, pipe, state, guard.reset_lease.evidence)
    module.recording(stack, context['sink'], recovery)
    lease = guard.reset_lease
    lease.empty_evidence = proof
    receiver = state['postcommit_current_receiver']
    prior = dict(issued=receiver.issued, released=receiver.released)
    trace = []
    try:
        clock['frame'] = I.RESET
        lease.perform(recovery, I.RESET, I.RESET/60)
        Q.KEPT['prefix_input_stack'].close()
        post = stack.enter_context(ExitStack())
        supplied = module.I.install(context['q'], post, pipe, context['pixels'], factory.types.parts.O, clock)
        for frame in I.POST:
            clock['frame'], cap.position = frame, frame
            context['collector'].collect_lean(cap, pipe, frame, 1, I.STRIDE, 60)
            trace.append(B.frame_state(factory, lease, frame))
        checked = verify(recovery, lease, factory, state, trace, prior)
        B.retire(lease, factory)
        result.update(prefix_updates=76, prefix_J_steps=result['J_steps'],
            updates=76+len(I.POST), J_steps=factory.provider.journal.steps, reset_continuation=checked)
        Q.KEPT['reset_continuation'] = checked
    finally:
        data = dict(trace=trace, recovery_rows=recovery.rows, lease_rows=lease.events,
                    pending=recovery.pending, error=recovery.error)
        with (state['output']/'EMPTY_RESET_CONTINUATION.json').open('x', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
