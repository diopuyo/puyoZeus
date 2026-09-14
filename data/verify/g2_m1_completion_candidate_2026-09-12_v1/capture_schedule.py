"""勝率を参照せず、事前固定窓内の最初の有資格採録を要求する有限状態。"""
from __future__ import annotations

from dataclasses import dataclass, replace

STRIDE = 2
MIN_GAP = 40
EARLIEST = (35370, 35410)
END = 36298


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError('capture_schedule:' + reason)


@dataclass(frozen=True)
class Schedule:
    last: int | None = None
    accepted: tuple[int, ...] = ()
    pending: int | None = None


def check(state: Schedule) -> None:
    require(type(state) is Schedule and type(state.accepted) is tuple
            and len(state.accepted) <= len(EARLIEST), 'state')
    require(state.last is None or type(state.last) is int and 0 <= state.last <= END
            and state.last % STRIDE == 0, 'clock_type')
    require(state.last is not None or not state.accepted and state.pending is None, 'initial_state')
    require(all(type(f) is int and f % STRIDE == 0 and EARLIEST[i] <= f <= state.last <= END
                for i, f in enumerate(state.accepted)), 'accepted_clock')
    require(all(b - a >= MIN_GAP for a, b in zip(state.accepted, state.accepted[1:])), 'accepted_gap')
    require(state.pending is None or (type(state.pending) is int and state.pending == state.last
                and len(state.accepted) < len(EARLIEST)), 'pending_clock')


def observe(state: Schedule, frame: int, reasons: tuple[str, ...]) -> tuple[Schedule, dict]:
    """reasonsは別の元資格判定器が出す正常HOLDのみ。内部例外をここへ変換しない。"""
    check(state)
    require(state.pending is None, 'previous_save_unfinished')
    require(type(frame) is int and 0 <= frame <= END and frame % STRIDE == 0, 'frame')
    require(frame < EARLIEST[0] if state.last is None else frame == state.last + STRIDE, 'continuous_clock')
    require(type(reasons) is tuple and all(type(v) is str and v for v in reasons)
            and len(set(reasons)) == len(reasons), 'reasons')
    following = replace(state, last=frame)
    index = len(state.accepted)
    due = None if index == len(EARLIEST) else max(EARLIEST[index],
        state.accepted[-1] + MIN_GAP if state.accepted else EARLIEST[0])
    action = 'COMPLETE' if due is None else 'WAIT'
    if due is not None and frame >= due and not reasons:
        following, action = replace(following, pending=frame), 'REQUEST'
    return following, dict(frame=frame, action=action, reasons=reasons, due=due,
        original_sentinel=frame in EARLIEST, quality_gate_clear=False)


def accept(state: Schedule, saved_frame: int) -> Schedule:
    """原保存が成功し、呼び手が原票を認証した後にのみ呼ぶ。要求だけで件数を増やさない。"""
    check(state)
    require(type(saved_frame) is int and state.pending == saved_frame == state.last, 'saved_frame')
    following = replace(state, accepted=state.accepted + (saved_frame,), pending=None)
    check(following)
    return following


def finish(state: Schedule) -> tuple[int, ...]:
    check(state)
    require(state.last == END and state.pending is None and len(state.accepted) == len(EARLIEST),
            'incomplete_coverage')
    return state.accepted
