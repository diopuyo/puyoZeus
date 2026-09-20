"""原80更新の後に、実v12のraw列を時刻だけ人工再配置するCPU入力。"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any
import tracking_loader as L

ROOT = Path(__file__).resolve().parent
ACTUAL_RESET = 35160
NATIVE_NEXT_OFFSET = 18
NATIVE_LAND_OFFSET = 20
BASIS_ONLY = os.environ.get('G2_TARGET_BASIS_ONLY') == '1'


def saved_raw() -> dict[int, Any]:
    path = ROOT.parent / 'video38_history_publication_probe_live_2026-09-11_v12/LIVE_EMPTY_RESET.json'
    data = json.loads(path.read_bytes())
    return {row['frame'] - ACTUAL_RESET: row['raw'] for row in data['recovery']
            if row['kind'] == 'reset_baseline_wait'}


def landed(raw: Any, B: Any, H: Any) -> Any:
    board = B.Board.from_dict({'grid': raw})
    # 人工正常対照の次手。元動画でこの手が置かれたという主張ではない。
    simulator = B.ChainSimulator(exclude_hidden_row_from_pop=True)
    grid = B.grid(board)
    candidates = H.enumerate_hypotheses(grid, (4, 5))
    for hypothesis in candidates:
        if hypothesis.touches_hidden:
            continue
        following = H.apply_hypothesis(grid, hypothesis)
        result = simulator.simulate(B.Board.from_dict({'grid': following}))
        if result.chain_count == 0 and result.final_board.get(1, 2) == 0:
            return [list(row) for row in following]
    raise ValueError('target_no_normal_visible_next_hand')


def install(original: Any, reset: int, q: Any, stack: Any, pipe: Any, pixels: Any,
            types: Any, clock: Any) -> Any:
    supplied = original(q, stack, pipe, pixels, types, clock)
    modules = L.modules()
    B, H = modules.belief, modules.mode.T.H
    rows = saved_raw()
    next_grid = rows[14] if BASIS_ONLY else landed(rows[14], B, H)
    helper = q.fixture_helpers(type(pipe._next_detector).detect_both)
    read, actual_next = pipe._reader.read_both_boards, helper.actual_next
    last: dict[str, Any] = {}
    def boards(image: Any, **kwargs: Any) -> Any:
        first, second = read(image, **kwargs)
        offset = clock['frame'] - reset
        raw = next_grid if offset >= NATIVE_LAND_OFFSET else rows[min(offset, 14)]
        result = type(first).from_dict({'grid': raw})
        last.update(first=result, second=second)
        return result, second
    def next_value(pair: Any, other: Any = None) -> Any:
        result = actual_next(pair, other)
        main = (4, 5) if BASIS_ONLY or clock['frame'] < reset + NATIVE_NEXT_OFFSET else (2, 2)
        return type(result)(type(result.p1)(*main, 3, 3), result.p2)
    modules.mode.M.patch(stack, pipe._reader, 'read_both_boards', boards)
    modules.mode.M.patch(stack, helper, 'actual_next', next_value)
    install_hsv(stack, pipe, modules.mode.M, last)
    return dict(original=supplied, actual_raw_source='live_v12', artificial_clock=True,
                artificial_next_hand=None if BASIS_ONLY else next_grid, basis_only=BASIS_ONLY,
                source_reset=ACTUAL_RESET, fixture_reset=reset, hsv_is_artificial_same_input=True)


def install_hsv(stack: Any, pipe: Any, mode: Any, last: dict[str, Any]) -> None:
    source = sys.modules[type(pipe).__module__]
    original = pipe._reader.read_board_hsv_only
    def hsv(image: Any, region: Any) -> Any:
        if not last:
            return original(image, region)
        if region == source.DEFAULT_P1_REGION:
            return last['first'].copy()
        if region == source.DEFAULT_P2_REGION:
            return last['second'].copy()
        raise ValueError('target_unknown_hsv_region')
    # 元fixtureのHSVは空Board固定。CPU人工入力の同意であり、独立画像認識ではない。
    mode.patch(stack, pipe._reader, 'read_board_hsv_only', hsv)
