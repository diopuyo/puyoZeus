"""同一落下の入場/内部/退出3観測だけで原3票guardを満たす私有候補。"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

REQUIRED_OBSERVATIONS = 3
FRAME_STRIDE = 2
FALL = 'tsumo_fall'
STABLE = 'stable'


@dataclass
class Tick:
    frame: int
    before: str
    after: str | None
    board: Any
    quiet: bool


def extended_history(ticks: list[Tick], history: list[Any]) -> list[Any] | None:
    """既存内部1票と同一物を含む3連続観測。多重計上/他遷移は拒否する。"""
    if len(ticks) != REQUIRED_OBSERVATIONS or len(history) != 1:
        return None
    entry, inside, exit_tick = ticks
    if (entry.before, entry.after, inside.before, inside.after, exit_tick.before) != (
            STABLE, FALL, FALL, FALL, FALL):
        return None
    if not all(tick.quiet for tick in ticks):
        return None
    if [tick.frame for tick in ticks] != [entry.frame + FRAME_STRIDE * i
                                        for i in range(REQUIRED_OBSERVATIONS)]:
        return None
    if not all(tick.board == entry.board for tick in ticks) or history[0] != inside.board:
        return None
    return [tick.board.copy() for tick in ticks]


class EndpointVotes:
    def __init__(self, machine: Any, allowed: Callable[[], bool]) -> None:
        self.machine, self.allowed = machine, allowed
        self.original_update = machine.update
        self.original_transition = machine._apply_transition
        self.ticks: deque[Tick] = deque(maxlen=REQUIRED_OBSERVATIONS)
        self.active: Tick | None = None
        self.rows: list[dict[str, Any]] = []

    def update(self, frame: int, signals: Any) -> Any:
        if not self.allowed():
            self.ticks.clear()
            return self.original_update(frame, signals)
        if self.active is not None:
            raise RuntimeError('endpoint_votes:reentrant_update')
        tick = Tick(frame, self.machine.context.state.value, None, signals.cnn_board.copy(),
                    signals.is_match_active and signals.effect_gate_window_active is False)
        self.ticks.append(tick)
        self.active = tick
        try:
            result = self.original_update(frame, signals)
            tick.after = result.state.value
            return result
        finally:
            self.active = None

    def transition(self, new_state: Any, signals: Any) -> Any:
        history = self.machine.context.non_stable_cnn_history
        if self.active is not None and new_state.value == STABLE:
            extra = extended_history(list(self.ticks), history)
            if extra is not None:
                # 閾値/確定盤面を触らず、原mergeへ実3callの票を一回ずつ渡す。
                self.machine.context.non_stable_cnn_history = extra
                self.rows.append(dict(frame=self.active.frame,
                    frames=[tick.frame for tick in self.ticks], votes=len(extra),
                    previous_votes=len(history), physical_certified=False))
        return self.original_transition(new_state, signals)


def install(stack: Any, machine: Any, allowed: Callable[[], bool]) -> EndpointVotes:
    """実factoryの資格callback/来歴接続は親担当。未接続部品としての入口。"""
    value = EndpointVotes(machine, allowed)
    for name in ('update', '_apply_transition'):
        prior_exists, prior = name in vars(machine), vars(machine).get(name)
        wrapper = value.update if name == 'update' else value.transition
        def restore(name: str = name, wrapper: Any = wrapper,
                    exists: bool = prior_exists, prior: Any = prior) -> None:
            if vars(machine).get(name) is not wrapper:
                raise RuntimeError('endpoint_votes:foreign_restore')
            if exists:
                setattr(machine, name, prior)
            else:
                delattr(machine, name)
        stack.callback(restore)
        setattr(machine, name, wrapper)
    return value
