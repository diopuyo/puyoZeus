"""人工OCR入力から原formula発火とhold終了を観測する。実Jや動画の合格ではない。"""
from __future__ import annotations
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any
import pytest
import test_tracker_input as F
import tracker_input as T

ROOT = Path(__file__).resolve().parent
FIRST, STRIDE, FIRE_INDEX, WINDOW, FPS = 34772, 2, 40, 4, 60
BEFORE_SCORE, POINTS, WHITE = 1274, 40, 255
FLAGS = dict(enable_chain_formula_read_verify=True, enable_formula_chain_count_update=True,
             enable_formula_step_interlude=True)


class Reader:
    def __init__(self, clock: dict[str, int], module: Any) -> None:
        self.clock, self.module = clock, module

    def firing(self) -> bool:
        return FIRE_INDEX <= self.clock['index'] < FIRE_INDEX + WINDOW

    def read_side(self, image: Any, side: str) -> Any:
        value = None if side == '1P' and self.firing() else BEFORE_SCORE
        if side == '1P' and self.clock['index'] >= FIRE_INDEX + WINDOW:
            value = BEFORE_SCORE + POINTS
        return value, 1.0

    def read_side_detail(self, image: Any, side: str) -> Any:
        value, confidence = self.read_side(image, side)
        return value, confidence, (None,) * 8, (1.0,) * 8

    def read_formula_side(self, image: Any, side: str, **kwargs: Any) -> Any:
        assert side == '1P' and self.firing(), 'formula_window'
        return self.module.FormulaReadResult(True, POINTS, 1, POINTS, 1.0)


def drive(real: Any, patch: Any) -> dict[str, Any]:
    from src import score_ocr as score
    pipe, _, _, image, _ = real
    module, clock = sys.modules[type(pipe).__module__], {'index': 0}
    packet = json.loads((ROOT.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v36/PROBABILISTIC_BASIS.jsonl').read_bytes())['state']
    before = module.Board.from_dict({'grid': packet['hidden_worlds'][0]['cells'] + packet['visible']})
    outcome = pipe._chain_tracker_1p._simulator.simulate(before)
    current, other, reader = {'board': before}, module.Board(), Reader(clock, score)
    patch.setattr(pipe._reader, 'read_both_boards', lambda *_a, **_k: (current['board'].copy(), other.copy()))
    for name, value in {'_score_ocr': reader, '_score_tracker_1p': score.ScoreTracker('1P', reader),
                        '_score_tracker_2p': score.ScoreTracker('2P', reader)}.items():
        assert getattr(pipe, name) is None
        patch.setattr(pipe, name, value)
    rows, seen = [], False
    limit = FIRE_INDEX + int((pipe.CHAIN_MAX_HOLD_SEC + module.CHAIN_FORMULA_READ_HOLD_SEC + 2) * FPS / STRIDE)
    for index in range(limit):
        clock['index'] = index
        frame, pixels = FIRST + index * STRIDE, image.copy()
        if index >= FIRE_INDEX + WINDOW: current['board'] = outcome.final_board
        if reader.firing():
            y1, y2, x1, x2 = score.SCORE_1P_REGION
            height, width = pixels.shape[:2]
            pixels[y1*height//1080:y2*height//1080, x1*width//1920:x2*width//1920] = WHITE
        pipe.update(frame, frame / FPS, pixels)
        origin, formula = pipe._active_chain_1p, pipe._formula_last_read_1p
        seen = seen or origin is not None
        rows.append(dict(frame=frame, time_sec=frame/FPS, origin=origin is not None,
            mechanism=None if origin is None else origin.mechanism,
            before_matches=None if origin is None else origin.before_board.to_dict() == before.to_dict(),
            formula_valid=None if formula is None else formula.valid,
            chain_count=None if origin is None else origin.chain_count,
            confirmed_matches=pipe._sm_1p.context.confirmed_board is not None and
                pipe._sm_1p.context.confirmed_board.to_dict() == outcome.final_board.to_dict(),
            chain_until=pipe._chain_until_1p, state=pipe._sm_1p.context.state.value))
        if seen and origin is None and rows[-1]['state'] == 'stable': break
    return dict(rows=rows, origin_seen=seen, origin_closed=seen and pipe._active_chain_1p is None,
        limit=limit, hold_seconds=module.CHAIN_FORMULA_READ_HOLD_SEC,
        exit_next_signal=pipe._enable_chain_exit_next_signal, artificial_single_world=True,
        actual_J=False, quality_gate_clear=False)


def main() -> int:
    started, evidence = time.monotonic(), {}
    path = ROOT.parent / 'g2_chain_firing_continuous_cpu_2026-09-10_v1/constructor_input.py'
    spec = importlib.util.spec_from_file_location('_g2_formula_constructor', path)
    constructor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constructor)
    with contextlib.contextmanager(F.F.frozen.__wrapped__)() as frozen:
        with pytest.MonkeyPatch.context() as patch:
            real = T.transport(constructor.transport(F.F.real.__wrapped__, FLAGS, evidence), evidence)
            with contextlib.contextmanager(real)(frozen, patch) as value:
                report = drive(value, patch)
    report.update(constructor=evidence, seconds=time.monotonic()-started)
    with (ROOT / 'FORMULA_PROBE_v2.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: report[key] for key in ('origin_seen', 'origin_closed', 'seconds')}))
    return int(not report['origin_closed'])


if __name__ == '__main__':
    raise SystemExit(main())
