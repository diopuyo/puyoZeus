"""退出票を含めず、遅延FALL直前からの3過去観測を原mergeへ渡す未接続候補。"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Any
import endpoint_votes as OLD

FPS = 60
WINDOW = 4


@dataclass
class Tick(OLD.Tick):
    clock: float = 0.0
    next_pair: tuple[int, int] | None = None
    history_tail_id: int | None = None


def past_history(ticks: list[Tick], history: list[Any]) -> list[Any] | None:
    """3過去票は退出rawを見ずに決める。現在rawとの照合は原mergeへ委ねる。"""
    if len(ticks) != WINDOW or len(history) != 1:
        return None
    previous, entry, inside, current = ticks
    phases = ((previous.before, previous.after), (entry.before, entry.after),
              (inside.before, inside.after), current.before)
    if phases != ((OLD.STABLE, OLD.STABLE), (OLD.STABLE, OLD.FALL),
                  (OLD.FALL, OLD.FALL), OLD.FALL):
        return None
    if not all(tick.quiet and type(tick.frame) is int and tick.clock == tick.frame / FPS
               for tick in ticks):
        return None
    if [tick.frame for tick in ticks] != [previous.frame + OLD.FRAME_STRIDE * i for i in range(WINDOW)]:
        return None
    if any(tick.next_pair != previous.next_pair for tick in ticks):
        return None
    if previous.board != entry.board or entry.board != inside.board:
        return None
    if id(history[0]) != inside.history_tail_id or history[0] != inside.board:
        return None
    return [tick.board.copy() for tick in ticks[:-1]]


def quiet(signals: Any) -> bool:
    flags = ('effect_gate_window_active', 'effect_visible', 'match_just_started', 'chain_counter_visible')
    return (signals.is_match_active is True and all(getattr(signals, key, None) is False for key in flags)
            and getattr(signals, 'chain_event', object()) is None)


class PriorVotes(OLD.EndpointVotes):
    def __init__(self, machine: Any, allowed: Any) -> None:
        super().__init__(machine, allowed)
        self.ticks: deque[Tick] = deque(maxlen=WINDOW)
        self.failure: BaseException | None = None

    def update(self, frame: int, signals: Any) -> Any:
        if self.failure is not None:
            raise RuntimeError('prior_votes:prior_failure') from self.failure
        if not self.allowed():
            self.ticks.clear()
            return self.original_update(frame, signals)
        if self.active is not None:
            raise RuntimeError('prior_votes:reentrant_update')
        tick = Tick(frame, self.machine.context.state.value, None, signals.cnn_board.copy(),
                    quiet(signals), signals.time_sec, signals.next_pair)
        self.ticks.append(tick)
        self.active = tick
        try:
            result = self.original_update(frame, signals)
            tick.after = result.state.value
            if tick.before == tick.after == OLD.FALL and result.non_stable_cnn_history:
                tick.history_tail_id = id(result.non_stable_cnn_history[-1])
            return result
        except BaseException as error:
            self.failure = error
            raise
        finally:
            self.active = None

    def transition(self, new_state: Any, signals: Any) -> Any:
        history = self.machine.context.non_stable_cnn_history
        extra = None
        if self.active is not None and new_state.value == OLD.STABLE:
            extra = past_history(list(self.ticks), history)
        if extra is not None:
            self.machine.context.non_stable_cnn_history = extra
        result = self.original_transition(new_state, signals)
        if extra is not None:
            self.rows.append(dict(frame=self.active.frame, frames=[tick.frame for tick in list(self.ticks)[:-1]],
                votes=len(extra), previous_votes=len(history), exit_used_as_vote=False,
                exit_agrees=(self.ticks[-1].board == self.ticks[-2].board), physical_certified=False))
        return result


def install(*args: Any, **kwargs: Any) -> PriorVotes:
    from types import FunctionType
    return FunctionType(OLD.install.__code__, dict(vars(OLD), EndpointVotes=PriorVotes))(*args, **kwargs)
