"""通常二手を挟む二回目連鎖の人工入力だけを原83更新入力へ追加する。"""
from __future__ import annotations
from typing import Any
from continuation_v1 import inputs as I

FIRST, LAST, STRIDE = 34772, 34990, 2
GG_LAND, YY_LAND, GG_NEXT = 34918, 34936, 34946
FIRE_LAND, MISSING_OCR, FINAL, NEXT = 34950, 34952, 34960, 34976
ROWS, COLS, BOTTOM, GREEN_COLUMN, YELLOW_COLUMN, OLD_COLUMN = 13, 6, 12, 3, 2, 5
FIRST_SCORE, FINAL_SCORE, GREEN, YELLOW, EMPTY = 140, 240, 3, 4, 0
GG, BB, RR, YY = (3, 3), (2, 2), (1, 1), (4, 4)
FRAMES = tuple(range(FIRST, LAST+STRIDE, STRIDE))


def board_at(original: Any, board: Any, frame: int) -> Any:
    value = original(board, frame)
    if frame >= GG_LAND:
        for row in (BOTTOM, BOTTOM-1): value.set(row, OLD_COLUMN, EMPTY)
        for row in (BOTTOM-1, BOTTOM-2): value.set(row, GREEN_COLUMN, GREEN)
    if frame >= YY_LAND:
        for row in (BOTTOM-1, BOTTOM-2): value.set(row, YELLOW_COLUMN, YELLOW)
    if frame >= FIRE_LAND:
        for row in (BOTTOM-3, BOTTOM-4): value.set(row, GREEN_COLUMN, GREEN)
    if frame >= FINAL:
        for row in range(BOTTOM-4, BOTTOM+1): value.set(row, GREEN_COLUMN, EMPTY)
    return value


def next_inputs(original: Any, patch: Any) -> Any:
    def inputs(q: Any, stack: Any, pipe: Any, pixels: Any, types: Any, clock: Any) -> Any:
        rows = original(q, stack, pipe, pixels, types, clock)
        helper = q.fixture_helpers(type(pipe._next_detector).detect_both)
        previous = helper.actual_next
        def value(pair: Any, other: Any = None) -> Any:
            result = previous(pair, other)
            if clock['frame'] < I.ADVANCE: return result
            first, second = (GG, BB) if clock['frame'] < GG_NEXT else (BB, RR) if clock['frame'] < NEXT else (RR, YY)
            return type(result)(type(result.p1)(*first, *second), result.p2)
        patch(stack, helper, 'actual_next', value)
        return rows
    return inputs


def install(stack: Any, source: Any, patch: Any, *, early_next: bool) -> None:
    assert early_next is False
    I.install(stack, source, patch, early_next=False)
    old_board, old_geometry, old_score = source.board_at, source.geometry, source.ArtificialScoreOcr.read_side
    def board(board: Any, frame: int) -> Any:
        return board_at(old_board, board, frame)
    def geometry(types: Any, invocation: Any, side: str) -> Any:
        if invocation.frame < GG_NEXT: return old_geometry(types, invocation, side)
        frame = invocation.frame
        number = {GG_NEXT: 7, NEXT: 8}.get(frame)
        event = None if number is None else types.MotionCandidate(number, frame, frame, f'artificial-repeat:{number}')
        quiet = (frame-STRIDE, frame) if frame in (GG_NEXT+STRIDE, NEXT+STRIDE) else None
        return types.MotionObservation(invocation, side, invocation.runtime.histories[side].epoch,
            'artificial-firing-segment', frame, invocation.time_sec, quiet, event)
    def score(self: Any, image: Any, side: str) -> Any:
        if side != '1P' or self.clock['frame'] < MISSING_OCR: return old_score(self, image, side)
        value = None if self.clock['frame'] < FINAL else FINAL_SCORE
        self.rows.append(dict(frame=self.clock['frame'], side=side, score=value))
        return value, 1.0
    patch(stack, source, 'board_at', board)
    patch(stack, source, 'geometry', geometry)
    patch(stack, source.ArtificialScoreOcr, 'read_side', score)
    patch(stack, source, 'inputs', next_inputs(source.inputs, patch))
    patch(stack, source, 'FRAMES', FRAMES)
