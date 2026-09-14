"""原Bridgeのpost-updateと原Session生成の間だけ、早期履歴の所有を接続する。"""
import json
from pathlib import Path
import sys
from typing import Any

VERIFY = Path(__file__).resolve().parent.parent
PUB = VERIFY / 'g2_belief_live_publication_2026-09-11_v1'
ASYNC = VERIFY / 'g2_async_projected_evaluation_2026-09-13_v1'
WRITER = VERIFY / 'g2_journal_writer_witness_repair_2026-09-11_v1'
ALIASES = ('_early_origin_context', '_early_origin_writer_contract', '_early_origin_writer_v2',
           '_early_origin_stream_witness', '_early_origin_reader', '_early_origin_history',
           '_early_probability_observation', '_early_probability_capture')


def modules(stack: Any, load: Any, *, probability: dict | None = None) -> tuple:
    if any(alias in sys.modules for alias in ALIASES): raise ValueError('early_runtime_foreign_alias')
    owned: dict = {}
    def release(kind: Any, body: Any, trace: Any) -> bool:
        changed = []
        for alias, value in owned.items():
            if sys.modules.get(alias) is value: sys.modules.pop(alias)
            else: changed.append(alias)
        owned.clear()
        if body is None and changed: raise ValueError('early_runtime_alias_changed:' + repr(changed))
        return False
    stack.push(release)
    def take(index: int, path: Path, injection: dict | None = None) -> Any:
        alias = ALIASES[index]
        try: return load(alias, path, injection)
        finally:
            value = sys.modules.get(alias)
            if value is not None and Path(value.__file__).resolve() == path.resolve(): owned[alias] = value
    context = take(0, PUB / 'journal_context.py')
    injection = {'journal_context': context}
    original = take(1, WRITER / 'writer_contract.py', injection)
    proof = take(2, WRITER / 'writer_contract_v2.py', injection | {'writer_contract': original})
    witness = take(3, WRITER / 'stream_witness.py', injection | {'writer_contract': proof})
    original_stream = witness.Stream
    class NamedStream(original_stream):
        @property
        def name(self) -> str:
            return self.witness.original.name
    witness.Stream = NamedStream
    def restore_stream(kind: Any, body: Any, trace: Any) -> bool:
        changed = witness.Stream is not NamedStream
        witness.Stream = original_stream
        if body is None and changed: raise ValueError('early_runtime_foreign_stream_class')
        return False
    stack.push(restore_stream)
    reader = take(4, ASYNC / 'journal_pair_reader.py', {'journal_witness': witness})
    history = take(5, ASYNC / 'early_origin_history.py')
    if probability is not None:
        if probability: raise ValueError('early_runtime_foreign_probability_modules')
        probability['observation'] = take(6, PUB / 'second_observation.py', injection)
        probability['capture'] = take(7, Path(__file__).resolve().parent / 'early_probability_capture.py')
    return witness, reader, history


class Runtime:
    def __init__(self, stack: Any, connection: Any, replace: Any, *, probability_enabled: bool = False) -> None:
        self.connection, self.replace = connection, replace
        self.history = self.bridge = None
        self.history_attempted = False
        self.closed = False
        self.probability_enabled = probability_enabled
        stack.callback(self.close)

    def close(self) -> None:
        self.closed = True
        self.connection = self.replace = self.history = self.bridge = None

    def require_history(self) -> Any:
        if self.closed or self.history is None or self.history.closed:
            raise ValueError('early_runtime_history_missing_or_closed')
        return self.history

    def install_binding(self, stack: Any) -> None:
        original = self.connection.binding.install
        def install(owner_stack: Any, bootstrap: Any, replace: Any, **kwargs: Any) -> Any:
            if 'history_getter' in kwargs: raise ValueError('early_runtime_foreign_history_getter')
            if 'fixed_reference_enabled' in kwargs: raise ValueError('early_runtime_foreign_fixed_reference')
            if 'root_probability_enabled' in kwargs: raise ValueError('early_runtime_foreign_root_probability')
            return original(owner_stack, bootstrap, replace, history_getter=self.require_history,
                            fixed_reference_enabled=True, root_probability_enabled=self.probability_enabled, **kwargs)
        self.replace(stack, self.connection.binding, 'install', install)

    def install_creator_boundary(self, stack: Any) -> None:
        original = self.connection.before_create
        def before_create(context: dict) -> None:
            history = self.require_history()
            if context['pipe'] is not history.pipe or context['factory'].provider.journal is not history.journal:
                raise ValueError('early_runtime_create_owner')
            history.seal(history.journal.history.frame)
            original(context)
        self.replace(stack, self.connection, 'before_create', before_create)
        audit = self.connection.audit
        def audited(session: Any, context: dict) -> None:
            history = self.require_history()
            if not history.transferred or session.journal.history.frame != history.last_frame:
                raise ValueError('early_runtime_prime_not_completed')
            audit(session, context)
            self.connection.binding.verify(self.connection.bound, require_fixed_reference=True)
            if self.probability_enabled:
                self.connection.binding.verify(self.connection.bound, require_root_probability=True)
            receipt = dict(recorded_first=history.frames[0], recorded_last=history.last_frame,
                available_frame=session.journal.history.frame, updates=history.count,
                source_sha256=history.digest.hexdigest(), live_history_handoff=True,
                probability_enabled=self.probability_enabled,
                probability_candidates=len(history.probability_inputs()),
                fixed_reference_enabled=True, physical_identity_verified=False, quality_gate_clear=False)
            with (session.state['output'] / 'EARLY_ORIGIN_CONNECTION.json').open('x') as stream:
                json.dump(receipt, stream)
        self.replace(stack, self.connection, 'audit', audited)

    def begin_history(self, bridge: Any, owner: Any, pipeline: Any) -> None:
        if self.closed or self.history_attempted or bridge is not self.bridge:
            raise ValueError('early_runtime_history_start_owner')
        self.history_attempted = True
        hook_stack = probability_owner(bridge, pipeline) if self.probability_enabled else None
        components = {} if self.probability_enabled else None
        witness, reader, history = modules(self.connection.active_binding_stack(), owner.bootstrap().load,
                                           probability=components)
        def probability_factory(scope: Any) -> Any:
            return components['capture'].Capture(scope, bridge.state['atomic_journal_observer'],
                bridge.state, components['observation'], self.replace, hook_stack=hook_stack)
        self.history = history.EarlyHistory(bridge.stack, bridge.state['atomic_journal_observer'],
            bridge.frames, bridge.state['output'], witness, reader,
            probability_factory=probability_factory if self.probability_enabled else None)

    def install_bridge(self, stack: Any, whole: Any, owner: Any) -> None:
        original = whole.W.Bridge
        def bridge(*args: Any, **kwargs: Any) -> Any:
            if self.closed or self.bridge is not None: raise ValueError('early_runtime_duplicate_bridge')
            value = original(*args, **kwargs)
            self.bridge = value
            original_caller = value.caller
            def caller(frame: Any) -> dict:
                values = original_caller(frame)
                if self.history is None: self.begin_history(value, owner, values['pipeline'])
                return values
            self.replace(value.stack, value, 'caller', caller)
            driver = value.consumer
            def completed(boundary: Any, frame: int) -> None:
                if boundary is not value: raise ValueError('early_runtime_bridge_owner')
                selected = self.require_history()
                if not selected.sealed: selected.observe(value.initial['pipeline'], frame)
                driver(boundary, frame)
            value.consumer = completed
            return value
        self.replace(stack, whole.W, 'Bridge', bridge)


def probability_owner(bridge: Any, pipeline: Any = None) -> Any:
    """後付け復帰hookと同じ所有stack以外へ退避しない。"""
    state = bridge.state
    context, arming = state['live_empty_reset_context'], state['live_empty_arming']
    if (context.state is not state or arming.state is not state or context.pipe is not arming.pipe
            or arming.callback.__self__ is not context or arming.stack is not context.stack
            or context.factory.provider.journal is not state['atomic_journal_observer']):
        raise ValueError('early_runtime_probability_hook_owner')
    if pipeline is not None and context.pipe is not pipeline:
        raise ValueError('early_runtime_probability_pipeline_owner')
    return context.stack
