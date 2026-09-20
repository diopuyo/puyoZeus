"""発火手までの人工入力。許可票・起源を作るのは元処理のみ。"""
from __future__ import annotations
from typing import Any

FRAMES = tuple(range(34772, 34850, 2))
FIRST, BASELINE, GY_LAND, GY_END = 34778, 34796, 34798, 34804
BP_LAND, BP_END, PP_LAND, PP_END, FIRE = 34810, 34822, 34828, 34840, 34844
GY, BP, PP, GG, YY = (4, 3), (2, 5), (5, 5), (3, 3), (4, 4)
BOTTOM, PURPLE_COL, SCORE, LEFT, RIGHT, WHITE = 12, 1, 40, 50, 2, 255


def board_at(board: Any, frame: int) -> Any:
    result = board.copy()
    if frame >= GY_LAND:
        result.set(BOTTOM, 2, GY[0])
        result.set(BOTTOM, 3, GY[1])
    if frame >= BP_LAND:
        result.set(BOTTOM, 0, BP[0])
        result.set(BOTTOM, PURPLE_COL, BP[1])
    if frame >= PP_LAND:
        for row in (11, 10):
            result.set(row, PURPLE_COL, PP[0])
    if frame >= FIRE:
        for row in (9, 8):
            result.set(row, PURPLE_COL, PP[0])
    return result


def geometry(types: Any, invocation: Any, side: str) -> Any:
    frame = invocation.frame
    number = {FIRST: 1, GY_END: 2, BP_END: 3, PP_END: 4}.get(frame)
    event = None if number is None else types.MotionCandidate(number, frame, frame,
        f'artificial-firing-geometry:{number}')
    quiet = (frame-2, frame) if frame in (FRAMES[0], FIRST+2, GY_END+2, BP_END+2, PP_END+2) else None
    return types.MotionObservation(invocation, side, invocation.runtime.histories[side].epoch,
        'artificial-firing-segment', frame, invocation.time_sec, quiet, event)


def inputs(q: Any, stack: Any, pipe: Any, pixels: Any, types: Any, clock: Any) -> list[Any]:
    helper = q.fixture_helpers(type(pipe._next_detector).detect_both)
    original, call, read = helper.actual_next, type(pixels).__call__, pipe._reader.read_both_boards
    rows: list[Any] = []
    def generated(pair: Any, other: Any = None) -> Any:
        value, frame = original(pair, other), clock['frame']
        first, dnext = ((GY, BP) if frame < FIRST else (BP, PP) if frame < GY_END
                       else (PP, PP) if frame < BP_END else (PP, GG) if frame < PP_END else (GG, YY))
        return type(value)(type(value.p1)(*first, *dnext), value.p2)
    def observed(self: Any, invocation: Any, side: str) -> Any:
        value = call(self, invocation, side)
        if self is not pixels or side != '1P':
            return value
        result = geometry(types, invocation, side)
        rows.append(dict(frame=invocation.frame, quiet=result.quiet_frames,
            candidate=None if result.candidate is None else vars(result.candidate), artificial=True))
        return result
    def boards(image: Any, **kwargs: Any) -> Any:
        first, second = read(image, **kwargs)
        return board_at(first, clock['frame']), second
    stack.callback(setattr, helper, 'actual_next', original)
    stack.callback(setattr, type(pixels), '__call__', call)
    stack.callback(setattr, pipe._reader, 'read_both_boards', read)
    helper.actual_next, type(pixels).__call__, pipe._reader.read_both_boards = generated, observed, boards
    return rows


class ArtificialScoreOcr:
    """OCR 型の入力代替。スコア累積・式判定は元クラスに委ねる。"""
    def __init__(self, clock: Any, module: Any) -> None:
        self.clock, self.module, self.rows = clock, module, []

    def read_side(self, image: Any, side: str) -> Any:
        value = None if side == '1P' and self.clock['frame'] >= FIRE else SCORE
        self.rows.append(dict(frame=self.clock['frame'], side=side, score=value))
        return value, 1.0

    def read_side_detail(self, image: Any, side: str) -> Any:
        value, confidence = self.read_side(image, side)
        return value, confidence, (None,)*8, (1.0,)*8

    def read_formula_side(self, image: Any, side: str, **kwargs: Any) -> Any:
        assert side == '1P' and self.clock['frame'] >= FIRE
        return self.module.FormulaReadResult(True, LEFT, RIGHT, LEFT*RIGHT, 1.0)


def score_input(stack: Any, pipe: Any, clock: Any) -> Any:
    from src import score_ocr
    reader = ArtificialScoreOcr(clock, score_ocr)
    values = {'_score_ocr': reader, '_score_tracker_1p': score_ocr.ScoreTracker('1P', reader),
              '_score_tracker_2p': score_ocr.ScoreTracker('2P', reader)}
    for name, value in values.items():
        original = getattr(pipe, name)
        assert original is None, 'original_fixture_score_input_not_empty'
        stack.callback(setattr, pipe, name, original)
        setattr(pipe, name, value)
    return reader


def paint_score(cap: Any, frame: int) -> None:
    from src.score_ocr import SCORE_1P_REGION
    if frame >= FIRE:
        y1, y2, x1, x2 = SCORE_1P_REGION
        height, width = cap.pixels.shape[:2]
        cap.pixels[y1*height//1080:y2*height//1080, x1*width//1920:x2*width//1920] = WHITE
