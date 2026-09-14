"""原prefix後のreset区間へ人工OCRを供給し、基準取得後に式と消去後入力を出す。"""
from __future__ import annotations
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
POST_COUNT, STRIDE, FIRE_OFFSET, WINDOW_FRAMES = 220, 2, 80, 8
BEFORE_SCORE, POINTS, WHITE = 1274, 40, 255
SOURCE = ROOT.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v36/PROBABILISTIC_BASIS.jsonl'


class Reader:
    def __init__(self, clock: Any, module: Any, reset: int) -> None:
        self.clock, self.module = clock, module
        self.reset = reset

    def firing(self) -> bool:
        return self.reset + FIRE_OFFSET <= self.clock['frame'] < self.reset + FIRE_OFFSET + WINDOW_FRAMES

    def read_side(self, image: Any, side: str) -> Any:
        if side == '1P' and self.firing(): return None, 1.0
        after = side == '1P' and self.clock['frame'] >= self.reset + FIRE_OFFSET + WINDOW_FRAMES
        return BEFORE_SCORE + (POINTS if after else 0), 1.0

    def read_side_detail(self, image: Any, side: str) -> Any:
        value, confidence = self.read_side(image, side)
        return value, confidence, (None,) * 8, (1.0,) * 8

    def read_formula_side(self, image: Any, side: str, **kwargs: Any) -> Any:
        assert side == '1P' and self.firing(), 'cascade_formula_window'
        return self.module.FormulaReadResult(True, POINTS, 1, POINTS, 1.0)


def extend(original: Any, reset: int) -> Any:
    def run(context: Any, result: Any) -> None:
        from src import score_ocr as score
        pipe, cap, clock, stack = (context[k] for k in ('pipe', 'cap', 'clock', 'stack'))
        assert 'read' not in vars(cap), 'cascade_cap_instance_read_not_supported'
        reader = Reader(clock, score, reset)
        owned = stack.enter_context(ExitStack())
        for name, value in {'_score_ocr': reader, '_score_tracker_1p': score.ScoreTracker('1P', reader),
                            '_score_tracker_2p': score.ScoreTracker('2P', reader)}.items():
            old = getattr(pipe, name)
            assert old is None, 'cascade_fixture_score_not_empty'
            setattr(pipe, name, value)
            owned.callback(setattr, pipe, name, old)
        previous = cap.read
        def read() -> Any:
            ok, image = previous()
            if ok and reader.firing():
                y1, y2, x1, x2 = score.SCORE_1P_REGION
                height, width = image.shape[:2]
                image[y1*height//1080:y2*height//1080, x1*width//1920:x2*width//1920] = WHITE
            return ok, image
        cap.read = read
        owned.callback(delattr, cap, 'read')
        original(context, result)
    return run


def install(previous: Any) -> Any:
    def inputs(original: Any, reset: int, q: Any, stack: Any, pipe: Any,
               pixels: Any, types: Any, clock: Any) -> Any:
        supplied = previous(original, reset, q, stack, pipe, pixels, types, clock)
        packet = json.loads(SOURCE.read_bytes())['state']
        module = sys.modules[type(pipe).__module__]
        artificial = module.Board.from_dict({'grid': packet['hidden_worlds'][0]['cells'] + packet['visible']})
        outcome = pipe._chain_tracker_1p._simulator.simulate(artificial)
        assert outcome.chain_count == 1
        read = pipe._reader.read_both_boards
        def boards(image: Any, **kwargs: Any) -> Any:
            first, second = read(image, **kwargs)
            if clock['frame'] >= reset + FIRE_OFFSET + WINDOW_FRAMES:
                return outcome.final_board.copy(), second
            return first, second
        pipe._reader.read_both_boards = boards
        stack.callback(setattr, pipe._reader, 'read_both_boards', read)
        old_hsv = pipe._reader.read_board_hsv_only
        def hsv(image: Any, region: Any) -> Any:
            if region == module.DEFAULT_P1_REGION and clock['frame'] >= reset + FIRE_OFFSET + WINDOW_FRAMES:
                return outcome.final_board.copy()
            return old_hsv(image, region)
        pipe._reader.read_board_hsv_only = hsv
        stack.callback(setattr, pipe._reader, 'read_board_hsv_only', old_hsv)
        supplied.update(cascade_artificial_single_world_observation=True, source=str(SOURCE),
                        formula_fire_frame=reset+FIRE_OFFSET, native_next_hand=False,
                        score_ocr_active_from_frame=reset, score_prefix_updates=False)
        return supplied
    return inputs
