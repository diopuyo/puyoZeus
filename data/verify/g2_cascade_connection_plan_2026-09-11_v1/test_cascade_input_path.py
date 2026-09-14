"""新score/cap輸送を原Pipelineで発火→STABLEまで通す。原Jの検収ではない。"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import pytest
import cascade_inputs as I
import tracker_input as T
import test_tracker_input as F

FIRST, FPS = 34932, 60
FLAGS = dict(enable_chain_formula_read_verify=True, enable_formula_chain_count_update=True,
             enable_formula_step_interlude=True)


class Capture:
    def __init__(self, image: Any) -> None:
        self.image = image

    def read(self) -> Any:
        return True, self.image.copy()


def drive(real: Any, patch: Any) -> dict[str, Any]:
    pipe, _, _, image, _ = real
    module = sys.modules[type(pipe).__module__]
    packet = json.loads(I.SOURCE.read_bytes())['state']
    board = module.Board.from_dict({'grid': packet['hidden_worlds'][0]['cells'] + packet['visible']})
    final = pipe._chain_tracker_1p._simulator.simulate(board).final_board
    clock, cap, rows, timeline = {'frame': FIRST}, Capture(image), [], []
    def body(context: Any, result: Any) -> None:
        for offset in range(0, I.POST_COUNT * I.STRIDE, I.STRIDE):
            clock['frame'] = FIRST + offset
            raw = final if offset >= I.FIRE_OFFSET + I.WINDOW_FRAMES else board
            patch.setattr(pipe._reader, 'read_both_boards', lambda *_a, **_k: (raw.copy(), module.Board()))
            _, pixels = cap.read()
            pipe.update(clock['frame'], clock['frame']/FPS, pixels)
            origin, formula = pipe._active_chain_1p, pipe._formula_last_read_1p
            key = (None if origin is None else id(origin), pipe._sm_1p.context.state.value)
            if not timeline or timeline[-1]['key'] != key:
                timeline.append(dict(key=key, frame=clock['frame'],
                    mechanism=None if origin is None else origin.mechanism,
                    trigger=None if origin is None else origin.trigger_sec,
                    score=None if origin is None else origin.total_score,
                    before=None if origin is None else origin.before_board.to_dict(),
                    confirmed=None if pipe._sm_1p.context.confirmed_board is None else
                        pipe._sm_1p.context.confirmed_board.to_dict()))
            if origin is not None:
                rows.append((origin.mechanism, origin.before_board.to_dict() == board.to_dict(),
                             None if formula is None else formula.valid))
    with ExitStack() as stack:
        I.extend(body, FIRST)(dict(pipe=pipe, cap=cap, clock=clock, stack=stack), {})
    assert 'read' not in vars(cap) and pipe._score_ocr is None
    assert pipe._score_tracker_1p is None and pipe._score_tracker_2p is None
    with (I.ROOT / 'DUPLICATE_ORIGIN_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(timeline=timeline, artificial=True, actual_J=False, quality_gate_clear=False), stream, indent=2)
    assert ('formula_read', True, True) in rows
    assert pipe._active_chain_1p is None and pipe._sm_1p.context.state.value == 'stable'
    assert pipe._sm_1p.context.confirmed_board.to_dict() == final.to_dict()
    return dict(rows=len(rows), actual_J=False)


def test_new_score_transport_original_pipeline() -> None:
    path = I.ROOT.parent / 'g2_chain_firing_continuous_cpu_2026-09-10_v1/constructor_input.py'
    spec = importlib.util.spec_from_file_location('_g2_test_cascade_constructor', path)
    constructor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constructor)
    evidence: dict[str, Any] = {}
    with contextmanager(F.F.frozen.__wrapped__)() as frozen:
        with pytest.MonkeyPatch.context() as patch:
            original = constructor.transport(F.F.real.__wrapped__, FLAGS, evidence)
            with contextmanager(T.transport(original, evidence))(frozen, patch) as real:
                assert drive(real, patch)['rows'] > 0
    assert evidence['constructor_restored'] and evidence['tracker_constructor_restored']
