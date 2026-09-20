"""最終Sessionの型を変えず、診断採録と終了保存を同じExitStackへ接続する。"""
from functools import wraps
import json
from typing import Any, Callable

STREAM_NAME = 'PROJECTED_ORIGIN_CAPTURE.jsonl'
CLOSE_NAME = 'PROJECTED_ORIGIN_CAPTURE_CLOSE.json'
MARKER = '_projected_origin_binding_owned'


class Binding:
    def __init__(self, stack: Any, session: Any, factory: Callable) -> None:
        self.session, self.stream, self.capture = session, None, None
        self.errors: list[BaseException] = []
        stack.push(self.close)  # open/consumer初期化の途中失敗も外側stackが回収する。
        self.stream = (session.state['output'] / STREAM_NAME).open('x', encoding='utf-8')
        self.capture = factory(session, self.stream)

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        frame = None if self.capture is None else self.capture.last_frame
        for item in (self.capture, self.stream):
            if item is not None:
                try: item.close()
                except BaseException as error: self.errors.append(error)
        packet = dict(last_frame=frame, original_error=None if body is None else repr(body),
            capture_closed=self.capture is None or self.capture.closed,
            stream_closed=self.stream is None or self.stream.closed,
            cleanup_errors=[repr(error) for error in self.errors], quality_gate_clear=False,
            live_hook_verified=False, future_fire_power_supply_authorized=False)
        self.session.projected_origin_cleanup = packet
        try:
            path = self.session.state['output'] / CLOSE_NAME
            with path.open('x', encoding='utf-8') as stream:
                json.dump(packet, stream, ensure_ascii=False, allow_nan=False)
        except BaseException as error:
            self.errors.append(error)
            self.session.projected_origin_cleanup_save_error = repr(error)
        self.capture = self.stream = None
        self.session = None
        if body is None and self.errors: raise self.errors[0]
        return False  # 元bodyがあれば終了時の別例外で置換しない。


def install_class(stack: Any, cls: type, factory: Callable, *, after_completed: Callable | None = None) -> None:
    """既存runtime_patchのSession型identity guardを保つ。重複装着は拒否する。"""
    if hasattr(cls, MARKER) or not all(key in vars(cls) for key in ('__init__', 'completed')):
        raise ValueError('origin_session_class_owner_or_duplicate')
    original_init, original_completed = cls.__init__, cls.completed

    @wraps(original_init)
    def initialize(self: Any, owner_stack: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, owner_stack, *args, **kwargs)
        self.projected_origin_binding = Binding(owner_stack, self, factory)

    @wraps(original_completed)
    def completed(self: Any, frame: int) -> Any:
        self.projected_origin_binding.capture.completed(frame)
        if after_completed is not None:
            after_completed(self.projected_origin_binding.capture, frame)
        return original_completed(self, frame)

    def restore(kind: Any, body: Any, trace: Any) -> bool:
        if cls.__init__ is not initialize or cls.completed is not completed:
            if body is None: raise ValueError('origin_session_foreign_method')
            return False  # 他所有の上書きは戻さない。
        cls.__init__, cls.completed = original_init, original_completed
        delattr(cls, MARKER)
        return False

    stack.push(restore)
    cls.__init__, cls.completed = initialize, completed
    setattr(cls, MARKER, True)
