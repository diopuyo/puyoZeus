"""通常ツモ退出の可視浮き一個だけを拒否する私有・既定無効の境界。"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import sys
from types import FrameType
from typing import Any, Iterator

ROWS, COLS, FIRST_VISIBLE, EMPTY = 13, 6, 1, 0
COLORS = frozenset(range(1, 6))
KNOWN = COLORS | {EMPTY, 9}
SM_SHA = '5cde1678718bbde42987034fde4f0029055e3ce5f305c5745cb8731f620f9770'
DETECTOR_SHA = '95d34c3c85839476e4d2957c7c419347971abe16d680e435eb44449522962b15'
UPDATE_LINE, TRANSITION_LINE = 1195, 1237
ACTIVE = False


def floating_single(before: Any, observed: Any) -> tuple[int, int] | None:
    """row0を読まず、可視既存値を保存した浮き一個という負例だけ返す。"""
    if before is None or observed is None:
        return None
    changes = []
    for row in range(FIRST_VISIBLE, ROWS):
        for col in range(COLS):
            old, new = int(before.get(row, col)), int(observed.get(row, col))
            if old not in KNOWN or new not in KNOWN:
                return None
            if old != new:
                if old != EMPTY or new not in COLORS:
                    return None
                changes.append((row, col))
    if len(changes) != 1:
        return None
    row, col = changes[0]
    if row < ROWS - 1 and int(observed.get(row + 1, col)) == EMPTY:
        return row, col
    return None


class Controller:
    def __init__(self, cls: type, detector_cls: type, observer: Any) -> None:
        self.cls, self.detector_cls, self.observer = cls, detector_cls, observer
        self.original = cls._apply_transition
        self.within = cls._update_within_current_state
        self.update_code = inspect.unwrap(cls.update).__code__
        self.state = sys.modules[cls.__module__].BoardState
        self.records: list[dict[str, Any]] = []
        self.sticky_error: str | None = None

    def adopted(self, caller: FrameType | None, sm: Any, state: Any, signals: Any) -> bool:
        """先勝ち原本updateの実localsへ束縛し、別detectorや直接呼出を除外。"""
        if caller is None or caller.f_code is not self.update_code:
            return False
        values = caller.f_locals
        detector = values.get('det')
        return (values.get('self') is sm and values.get('signals') is signals
            and values.get('res') is state and values.get('new_state') is state
            and type(detector) is self.detector_cls
            and any(detector is item for item in sm._detectors))

    def transition(self, sm: Any, state: Any, signals: Any, caller: FrameType | None) -> None:
        try:
            ctx = sm.context
            target = ctx.state is self.state.TSUMO_FALL and state is self.state.STABLE
            target = target and not signals.slide_motion and not signals.placement_validated
            cell = floating_single(ctx.confirmed_board, signals.cnn_board) if target else None
            if cell is None or not self.adopted(caller, sm, state, signals):
                return self.original(sm, state, signals)
            self.within(sm, signals)
            row = {'frame_idx': ctx.frame_idx, 'time_sec': ctx.time_sec,
                'reason': 'ordinary_tsumo_visible_floating_single', 'cell': list(cell),
                'within_calls': 1, 'transition_calls': 0, 'row0_read': False,
                'detector_return_modified': False, 'state_after': ctx.state.value,
                'quality_gate_clear': False}
            self.records.append(row)
            if self.observer is not None:
                self.observer(dict(row))
        except BaseException as exc:
            self.sticky_error = type(exc).__name__ + ':' + str(exc)
            raise


def validate(cls: type, detector_cls: type) -> None:
    for target, expected in ((cls, SM_SHA), (detector_cls, DETECTOR_SHA)):
        path = Path(inspect.getfile(target)).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('unfrozen_source')
    path = str(Path(inspect.getfile(cls)).resolve())
    for name, line in (('update', UPDATE_LINE), ('_apply_transition', TRANSITION_LINE)):
        function = inspect.unwrap(getattr(cls, name))
        if function.__code__.co_filename != path or function.__code__.co_firstlineno != line:
            raise RuntimeError('unfrozen_method:' + name)


@contextmanager
def install(cls: type, detector_cls: type, enabled: bool = False,
            observer: Any = None) -> Iterator[Controller | None]:
    """保存・復元のみ。既定無効時はクラスにもインスタンスにも触れない。"""
    global ACTIVE
    if not enabled:
        yield None
        return
    if ACTIVE:
        raise RuntimeError('nested_install')
    validate(cls, detector_cls)
    control = Controller(cls, detector_cls, observer)
    original = cls._apply_transition
    def wrapped(sm: Any, new_state: Any, signals: Any) -> None:
        frame = inspect.currentframe()
        try:
            control.transition(sm, new_state, signals, frame.f_back)
        finally:
            del frame
    ACTIVE = True
    cls._apply_transition = wrapped
    try:
        yield control
    finally:
        cls._apply_transition = original
        ACTIVE = False
