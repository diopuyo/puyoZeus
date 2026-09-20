"""実終了入口で旧構造・確率保存・参照復元をまとめる。本番権限は付与しない。"""
from __future__ import annotations
from dataclasses import asdict
from types import FunctionType, SimpleNamespace as N
from typing import Any
import probability_owner as OWNER
import probability_boundary as BOUNDARY
import probability_world as WORLD
import probability_saved as SAVED
import probability_stage as STAGE

KEY = 'live_probability_finalizer'


def clone(function: Any, **changes: Any) -> Any:
    value = FunctionType(function.__code__, dict(function.__globals__, **changes),
                         function.__name__, function.__defaults__, function.__closure__)
    value.__kwdefaults__ = function.__kwdefaults__
    return value


def prepared(original: Any, factory: Any, state: Any, lease: Any) -> dict[str, Any]:
    old = original.OLD  # 採用済みlive_finish_v2の追加guardも維持する。
    mode = state['probabilistic_tracking_mode']
    c = state['probabilistic_basis_connection']
    retired = OWNER.verify(old.RETIRED, state, lease, factory)
    world = WORLD.verify(old.WORLD, factory, state, lease)
    private = None
    journal = original.read(state['output'], 'atomic_journal.jsonl')
    full = original.read(state['output'], 'directional_history.jsonl')
    rows = [row for row in full if 'decision' in row]
    if factory.controller.private_suffix_completion_rows:
        def owner(*args: Any) -> bool:
            return OWNER.retired_owner(old.RETIRED, state, *args)
        retained = N(modules=state[OWNER.ARCHIVE_KEY].modules)
        base = clone(old.private_samecall, RETIRED=N(retired_owner=owner), sys=retained)
        function = clone(original.private_samecall, OLD=N(private_samecall=base), sys=retained)
        private = function(factory, lease, journal, rows, state['private_suffix_modules'].evidence)
    boundary = BOUNDARY.check(state['postcommit_publication_consumer'].rows, full, journal,
        lease.recovery.rows, original.read(state['output'], 'PROBABILISTIC_TRACKING.jsonl'),
        history_first=lease.archive.binding.owner.state.baseline_through.frame,
        end_frame=mode.native.last_frame, proof_frame=lease.empty_evidence.frame,
        basis_frame=mode.activation['frame'], scope=c.binding.scope)
    checked_boundary = clone(original.BOUNDARY.check,
        B=N(check=lambda *args: boundary | dict(prior_issued_frames=boundary['issued_frames'])))
    boundary = checked_boundary(state['postcommit_publication_consumer'].rows, full, journal,
        lease.recovery.rows, old_owner_scope=asdict(lease.archive.binding.owner.state.scope))
    # 実Connection/Recorderを生成したモジュールの元decoder/extractを保持して使う。
    serializer = type(c).__init__.__globals__['S']
    native = N(extract=type(mode.native).observe.__globals__['extract'])
    probability = SAVED.verify(state, serializer, native)
    return dict(retired=retired, world=world, private=private, boundary=boundary, probability=probability)


def evaluate(original: Any, factory_context: Any, goals: Any, rows: Any, legal: Any,
             output: Any, state: Any) -> dict[str, Any]:
    context = state['live_empty_reset_context']
    assert context.proof is not None and context.error is None
    assert context.recovery.error is None and context.recovery.failure is None
    assert context.recovery.pending is None and context.recovery.baseline_count == 0
    factory = factory_context(state)
    assert factory is context.factory and factory.controller.legal is legal and output == state['output']
    for refs in (state['rolling_prefix_references'], factory.controller.empty_runtime_references):
        assert refs['installed'] and refs['closed'] and refs['restored']
    status = original.read(output, 'LIVE_EMPTY_RESET.json')
    assert status['error'] is None and status['baseline_recovered'] is False
    lease = state['repeat_scope_guard'].reset_lease
    report = prepared(original, factory, state, lease)
    structure = STAGE.verify(original.OLD, goals, rows, legal, output, state, report['boundary'],
                             firing_rows=state['repeated_firing_constructor']['rows'])
    return dict(probability_validation=report, structure=structure, runtime_finalization_allowed=True,
                full_probability_finalizer_verified=True, physical_certified=False,
                production_permission=False, quality_gate_clear=False)


def install(stack: Any, main: Any, original: Any, factory_context: Any) -> None:
    target = main.__globals__['Q'].FINAL
    previous = target.evaluate
    def wrapped(goals: Any, rows: Any, legal: Any, output: Any, state: Any) -> Any:
        if not OWNER.tracking_selected(state):
            return previous(goals, rows, legal, output, state)
        assert KEY not in state, 'duplicate_probability_finalize'
        state[KEY] = dict(stage='entered', error=None)
        try:
            result = evaluate(original, factory_context, goals, rows, legal, output, state)
            main.__globals__['K'].write(output / 'LIVE_PROBABILITY_FINAL.json', result)
            state[KEY]['stage'] = 'saved'
            return result
        except BaseException as error:
            state[KEY]['error'] = repr(error)
            try:
                main.__globals__['K'].write(output / 'LIVE_PROBABILITY_FINAL_FAILURE.json', state[KEY])
            except BaseException as save_error:
                state[KEY]['failure_save_error'] = repr(save_error)
            raise
    def restore() -> None:
        assert target.evaluate is wrapped, 'probability_finalize_restore_binding'
        target.evaluate = previous
    stack.callback(restore)
    target.evaluate = wrapped
