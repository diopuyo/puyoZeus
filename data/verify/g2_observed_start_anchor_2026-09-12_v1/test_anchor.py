"""原full collectで開始条件・認識遅延・失効・保存/復元を確認する人工CPU。"""
from __future__ import annotations
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
from typing import Any
import pytest
import anchor as A

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_producer_time_capture_2026-09-12_v1'))
import test_capture as T
real = T.real
RISE, QUALIFIED, LAST = 60, 90, 96


def second_rise(c: Any) -> int:
    return RISE + math.ceil(c.GAME_BOUNDARY_DEBOUNCE_SEC * T.C.FPS) + T.C.STRIDE


def pipeline(c: Any, mode: str) -> Any:
    pipe = T.T.pipeline(c)
    original = pipe.update
    pipe._active_chain_1p = object() if mode == 'active_chain' else None
    pipe._active_chain_2p = None
    if mode == 'missing':
        del pipe._active_chain_1p
    def update(frame_idx: int, time_sec: float, frame: Any) -> Any:
        result = original(frame_idx, time_sec, frame)
        result.is_match_active = mode == 'no_boundary' or frame_idx >= RISE
        if mode in ('next_boundary', 'debounced'):
            second = second_rise(c) if mode == 'next_boundary' else 300
            result.is_match_active = RISE <= frame_idx < 240 or frame_idx >= second
        empty = mode not in ('or_only',) and not (mode == 'delayed' and frame_idx < 94)
        if mode == 'good' and frame_idx >= 94 or mode in ('next_boundary', 'debounced') and frame_idx >= 240:
            empty = False
        grid = T.T.np.zeros((13, 6), dtype=T.T.np.uint8)
        if not empty:
            grid[12, 0] = 1
        for side in (result.p1, result.p2):
            side.confirmed_board = c.Board.from_list(grid.tolist())
            side.board_provenance = 'observed'
            side.score = None if mode == 'unknown_score' else 0
            if mode == 'delayed' and frame_idx < 94:
                side.score = 50
        return result
    pipe.update = update
    return pipe


def run(real: Any, output: Path, mode: str, end: int = LAST) -> dict[str, Any]:
    c, b = real
    output.mkdir(exist_ok=False)
    state, pipe = {'output': output}, pipeline(c, mode)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(b.O, 'FIRST', 0)
        patch.setattr(b.O, 'LAST', end)
        with ExitStack() as stack:
            b.install(stack, c, None, state, enabled=True)
            tail = T.T.M.install(stack, state)
            loop = T.U.build(c, b.O.SOURCE, overrides=T.L.settings())
            stack.callback(loop.generator.close)
            original = loop.collect_lean
            loop.RecognitionPipeline = type(pipe)
            capture = A.install(stack, loop, tail, {'source_id': 'video_38', 'run_id': output.name})
            cap = T.N(read=lambda: (True, T.T.np.zeros((1080, 1920, 3), dtype=T.T.np.uint8)))
            for frame in range(0, end + T.C.STRIDE, T.C.STRIDE):
                try:
                    loop.collect_lean(cap, pipe, frame, 1, T.C.STRIDE, T.C.FPS)
                except AttributeError as error:
                    assert error is capture.error is loop.error
                    with pytest.raises(AttributeError) as again:
                        capture.snapshot()
                    assert again.value is error
                    raise
            saved = capture.snapshot()
        assert capture.closed and loop.collect_lean is original
        b.finish(state)
        b.verify(output)
    with (output / 'START_CAPTURE.json').open('x', encoding='utf-8') as stream:
        json.dump(saved, stream, ensure_ascii=False, indent=2)
    return saved


@pytest.mark.parametrize('mode,qualification', [('good', 90), ('delayed', 94), ('or_only', None),
    ('unknown_score', None), ('active_chain', None), ('no_boundary', None)])
def test_original_start_conditions(real: Any, tmp_path: Path, mode: str, qualification: int | None) -> None:
    saved = run(real, tmp_path / mode, mode)
    assert saved['start_observation_eligible'] == (qualification is not None)
    assert saved['game_anchor_qualified'] is False and saved['live_qualified'] is False
    if qualification is not None:
        anchor = saved['start_anchor']
        assert anchor['qualification_frame'] == qualification
        assert anchor['candidate']['boundary_available_frame'] == QUALIFIED
        assert all(r['frame_idx'] == qualification for r in anchor['metadata'])
        value = {k: v for k, v in anchor.items() if k not in ('evidence_sha256', 'classification')}
        assert A.digest(value) == anchor['evidence_sha256']
    if mode == 'delayed':
        assert any(d['frame'] == QUALIFIED and not d['eligible'] for d in saved['start_decisions'])


def test_next_advance_retires_anchor(real: Any, tmp_path: Path) -> None:
    second = second_rise(real[0]) + math.ceil(real[0].BOUNDARY_VISUAL_RISE_PERSIST_SEC * T.C.FPS)
    saved = run(real, tmp_path / 'next', 'next_boundary', second + T.C.STRIDE)
    assert any(d['eligible'] and d['frame'] == QUALIFIED for d in saved['start_decisions'])
    assert saved['start_anchor'] is None and saved['observed_game_idx'] == 2
    assert saved['start_candidate']['boundary_available_frame'] == second


def test_debounced_visual_only_keeps_anchor(real: Any, tmp_path: Path) -> None:
    saved = run(real, tmp_path / 'debounced', 'debounced', 332)
    assert saved['observed_game_idx'] == 1 and saved['start_anchor']['qualification_frame'] == QUALIFIED
    assert sum(e['list'] == 'advance_times' for e in saved['boundary_events']) == 1
    assert sum(e['list'] == 'visual_rise_times' for e in saved['boundary_events']) == 2


def test_missing_contract_sticky(real: Any, tmp_path: Path) -> None:
    with pytest.raises(AttributeError):
        run(real, tmp_path / 'missing', 'missing')
