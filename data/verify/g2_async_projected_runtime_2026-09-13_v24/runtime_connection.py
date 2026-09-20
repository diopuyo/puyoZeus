"""原creator/createを一回ずつ呼び、同stack内の実Session接続を必須検査する。"""
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator

MODE_KEY = 'probabilistic_tracking_mode'
RECEIPT = 'PROJECTED_RUNTIME_CONNECTION.json'
LATE = Path(__file__).resolve().parent.parent / 'g2_m1_loader_lifetime_2026-09-13_v1/late_loader.py'


class Connection:
    def __init__(self, stack: Any, lease: Any, binding: Any, replace: Any) -> None:
        self.lease, self.binding, self.replace = lease, binding, replace
        self.owner = self.bound = self.borrowed = None
        self.creator_called = self.create_called = self.closed = False
        self.receipt: dict | None = None
        self.binding_scope: Any = None
        self.environment_called = False
        stack.callback(self.close)

    def close(self) -> None:
        self.closed = True
        self.owner = self.bound = self.borrowed = None
        self.lease = self.binding = self.replace = None
        self.binding_scope = None

    def active_binding_stack(self) -> Any:
        if self.closed or self.binding_scope is None:
            raise ValueError('a24_bind_outside_binding_scope')
        return self.binding_scope

    def install_environment(self, stack: Any, selected: Any) -> None:
        session = selected.__globals__['S']
        original = session.configured
        @contextmanager
        def configured() -> Iterator[Any]:
            if self.closed or self.environment_called:
                raise ValueError('a24_repeated_or_closed_environment')
            self.environment_called = True
            try:
                with original() as env:
                    with ExitStack() as scope:
                        self.binding_scope = scope
                        yield env
            finally:
                self.binding_scope = None
        self.replace(stack, session, 'configured', configured)

    def install_bootstrap(self, stack: Any, owner: Any) -> None:
        original = owner.bootstrap
        def bootstrap() -> Any:
            if self.closed: raise ValueError('a24_connection_closed')
            value = original()
            if self.owner is None:
                self.owner = value
                self.bound = self.binding.install(stack, value, self.replace, prefix_lease=self.lease,
                                                  binding_stack=self.active_binding_stack)
            elif self.owner is not value:
                raise ValueError('a24_bootstrap_owner_changed')
            return value
        self.replace(stack, owner, 'bootstrap', bootstrap)

    def before_create(self, context: dict) -> None:
        if self.closed or self.create_called:
            raise ValueError('a24_repeated_or_closed_create')
        self.active_binding_stack()
        if self.bound is None or self.bound['patch'] is None:
            raise ValueError('a24_patch_not_hooked')
        live, adapter = self.lease.current()
        first = context['state'][MODE_KEY]
        if type(first) is not live.module.Mode:
            raise ValueError('a24_first_mode_type')
        self.lease.mode_open(first)
        self.borrowed = (live, adapter, live.module.Mode, live.module.Mode.close)
        self.create_called = True

    def audit(self, session: Any, context: dict) -> None:
        self.binding.verify(self.bound, require_instance=True, require_projected_input=True)
        selected = self.bound['binding']
        if type(session) is not selected['cls'] or session.state is not context['state']:
            raise ValueError('a24_session_instance_owner')
        if session.pipe is not context['pipe'] or session.factory is not context['factory']:
            raise ValueError('a24_session_context_owner')
        live, adapter = self.lease.current()
        actual = (live, adapter, live.module.Mode, live.module.Mode.close)
        if any(old is not new for old, new in zip(self.borrowed, actual, strict=True)):
            raise ValueError('a24_lease_owner_changed')
        self.lease.mode_open(session.state[MODE_KEY])
        if session.projected_origin_binding.capture.projected_input is None:
            raise ValueError('a24_callback_missing')
        receipt = dict(construction_frame=session.journal.history.frame,
            pipe_object_id=id(session.pipe), live_object_id=id(live),
            session_class_verified=True, lease_live_captured=True, projected_input_enabled=True,
            quality_gate_clear=False, actual_video_verified=False, source_producer_authorized=False)
        with (session.state['output'] / RECEIPT).open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, ensure_ascii=False, allow_nan=False)
        self.receipt = receipt
        session.state['projected_runtime_connection'] = dict(receipt)

    def install_creator(self, stack: Any, dependencies: Any) -> None:
        original = dependencies.session_creator
        if Path(original.__code__.co_filename).resolve() != LATE:
            raise ValueError('a24_creator_source')
        def creator(load: Any, frames: tuple[int, ...]) -> Any:
            if self.closed or self.creator_called:
                raise ValueError('a24_repeated_or_closed_creator')
            self.creator_called = True
            create = original(load, frames)
            def wrapped(inner: Any, context: dict) -> Any:
                self.before_create(context)
                session = create(inner, context)
                self.audit(session, context)
                return session
            return wrapped
        self.replace(stack, dependencies, 'session_creator', creator)
