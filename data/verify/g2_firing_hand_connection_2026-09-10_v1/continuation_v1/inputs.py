"""発火後の実観測入力を人工的に追加する。S/SM/current/FIFOへは書かない。"""
from __future__ import annotations
from typing import Any

FIRST, LAST, STRIDE, FPS = 34772, 34936, 2, 60
FINAL, EARLY_NEXT, LATE_NEXT, LANDED, ADVANCE = 34852, 34850, 34910, 34918, 34928
PURPLE_COLUMN, TOP_FIRED, BOTTOM, NEW_COLUMN = 1, 8, 12, 5
ERASE_SCORE, BASE_SCORE = 100, 40
GG, YY, BB, RR = (3, 3), (4, 4), (2, 2), (1, 1)
FRAMES = tuple(range(FIRST, LAST+STRIDE, STRIDE))


def install(stack: Any, source: Any, patch: Any, *, early_next: bool) -> None:
    old_board, old_geometry = source.board_at, source.geometry
    old_inputs, old_score = source.inputs, source.ArtificialScoreOcr.read_side
    next_frame = EARLY_NEXT if early_next else LATE_NEXT
    def board_at(board: Any, frame: int) -> Any:
        value = old_board(board, frame)
        if frame >= FINAL:
            for row in range(TOP_FIRED, BOTTOM+1):
                value.set(row, PURPLE_COLUMN, 0)
        if frame >= LANDED:
            for row in (BOTTOM, BOTTOM-1):
                value.set(row, NEW_COLUMN, GG[0])
        return value
    def geometry(types: Any, invocation: Any, side: str) -> Any:
        if invocation.frame < next_frame:
            return old_geometry(types, invocation, side)
        number = {next_frame:5, ADVANCE:6}.get(invocation.frame)
        event = None if number is None else types.MotionCandidate(number, invocation.frame,
            invocation.frame, f'artificial-firing-continuation:{number}')
        quiet = ((invocation.frame-STRIDE, invocation.frame) if invocation.frame in
                 (next_frame+STRIDE, ADVANCE+STRIDE) else None)
        return types.MotionObservation(invocation, side, invocation.runtime.histories[side].epoch,
            'artificial-firing-segment', invocation.frame, invocation.time_sec, quiet, event)
    def score(self: Any, image: Any, side: str) -> Any:
        if side != '1P' or self.clock['frame'] < FINAL:
            return old_score(self, image, side)
        value = BASE_SCORE+ERASE_SCORE
        self.rows.append(dict(frame=self.clock['frame'], side=side, score=value))
        return value, 1.0
    patch(stack, source, 'board_at', board_at)
    patch(stack, source, 'geometry', geometry)
    patch(stack, source.ArtificialScoreOcr, 'read_side', score)
    patch(stack, source, 'inputs', inputs(old_inputs, next_frame, patch))
    patch(stack, source, 'FRAMES', FRAMES)


def inputs(original: Any, next_frame: int, patch: Any) -> Any:
    def installed(q: Any, stack: Any, pipe: Any, pixels: Any, types: Any, clock: Any) -> Any:
        rows = original(q, stack, pipe, pixels, types, clock)
        helper = q.fixture_helpers(type(pipe._next_detector).detect_both)
        generated = helper.actual_next
        def next_value(pair: Any, other: Any = None) -> Any:
            value = generated(pair, other)
            if clock['frame'] < next_frame:
                return value
            first, following = (YY, BB) if clock['frame'] < ADVANCE else (BB, RR)
            return type(value)(type(value.p1)(*first, *following), value.p2)
        patch(stack, helper, 'actual_next', next_value)
        return rows
    return installed
